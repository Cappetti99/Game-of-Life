/**
 * Game of Life CUDA Implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// BLOCK_SIZE is passed via command line arguments during compilation
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#define DEFAULT_SEED 42

#define TILE_SIZE (BLOCK_SIZE + 2)
#define SHARED_MEM_SIZE (TILE_SIZE * (TILE_SIZE + 1))



#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * Kernel to initialize the grid with random values.
 * Uses cuRAND to generate random states for each cell.
 */
__global__ void initGridKernel(unsigned char* grid, int width, int height, 
                                unsigned long seed) {
    // Calculate global thread coordinates
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        curandState state;
        curand_init(seed, idx, 0, &state);
        grid[idx] = (curand_uniform(&state) < 0.3f) ? 1 : 0;
    }
}

/**
 * Optimized Game of Life Kernel using Shared Memory Tiling.
 * 
 * Strategy:
 * 1. Each block loads a tile of the grid into Shared Memory.
 * 2. Shared Memory includes a "halo" (padding) to store neighbors from adjacent blocks.
 * 3. Threads cooperatively load the central tile and the halo regions.
 * 4. __ldg() intrinsic is used to utilize the Read-Only Data Cache.
 * 5. Padding in Shared Memory prevents bank conflicts.
 */
__global__ void gameOfLifeKernel(const unsigned char* currentGrid,
                                  unsigned char* nextGrid,
                                  int width, int height) {
    // Shared memory tile with halo (border) and padding to avoid bank conflicts (+1)
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE + 1];
    
    // Global coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local coordinates within the tile (offset by 1 for halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // 1. Load the central tile data
    if (x < width && y < height) {
        // Use __ldg to force load through the read-only cache (texture cache)
        tile[ty][tx] = __ldg(&currentGrid[y * width + x]);
    }
    
    // 2. Load Halo Cells (Ghost Cells)
    // We handle toroidal boundary conditions (wrapping edges)
    
    // Left halo
    if (threadIdx.x == 0) {
        int nx = (x - 1 + width) % width;
        if (y < height) tile[ty][0] = __ldg(&currentGrid[y * width + nx]);
    }
    // Right halo
    if (threadIdx.x == blockDim.x - 1 || x == width - 1) {
        int nx = (x + 1) % width;
        if (y < height) tile[ty][tx + 1] = __ldg(&currentGrid[y * width + nx]);
    }
    // Top halo
    if (threadIdx.y == 0) {
        int ny = (y - 1 + height) % height;
        if (x < width) tile[0][tx] = __ldg(&currentGrid[ny * width + x]);
    }
    // Bottom halo
    if (threadIdx.y == blockDim.y - 1 || y == height - 1) {
        int ny = (y + 1) % height;
        if (x < width) tile[ty + 1][tx] = __ldg(&currentGrid[ny * width + x]);
    }
    
    // Corner halo cells
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int nx = (x - 1 + width) % width;
        int ny = (y - 1 + height) % height;
        tile[0][0] = __ldg(&currentGrid[ny * width + nx]);
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        int nx = (x + 1) % width;
        int ny = (y - 1 + height) % height;
        tile[0][tx + 1] = __ldg(&currentGrid[ny * width + nx]);
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        int nx = (x - 1 + width) % width;
        int ny = (y + 1) % height;
        tile[ty + 1][0] = __ldg(&currentGrid[ny * width + nx]);
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        int nx = (x + 1) % width;
        int ny = (y + 1) % height;
        tile[ty + 1][tx + 1] = __ldg(&currentGrid[ny * width + nx]);
    }
    
    // Ensure all threads have finished loading shared memory before computation
    __syncthreads();
    
    // 3. Compute next state using Shared Memory
    if (x < width && y < height) {
        int neighbors = 0;
        
        // Sum neighbors from the 3x3 window stored in shared memory
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                neighbors += tile[ty + dy][tx + dx];
            }
        }
        
        // Apply Conway's Game of Life rules
        unsigned char current = tile[ty][tx];
        nextGrid[y * width + x] = (current == 1) 
            ? ((neighbors == 2 || neighbors == 3) ? 1 : 0)
            : ((neighbors == 3) ? 1 : 0);
    }
}



/**
 * Main simulation runner. Use "Ping-Pong" buffering strategy (swapping pointers)
 * to avoid memory copies between generations.
 */
float runSimulation(int width, int height, int generations, unsigned long seed) {
    size_t gridSize = width * height * sizeof(unsigned char);
    
    unsigned char *d_gridA, *d_gridB;
    CUDA_CHECK(cudaMalloc(&d_gridA, gridSize));
    CUDA_CHECK(cudaMalloc(&d_gridB, gridSize));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    initGridKernel<<<gridDim, blockDim>>>(d_gridA, width, height, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    unsigned char *current = d_gridA, *next = d_gridB;
    for (int gen = 0; gen < generations; gen++) {
        gameOfLifeKernel<<<gridDim, blockDim>>>(current, next, width, height);
        CUDA_CHECK(cudaGetLastError());
        unsigned char* temp = current; current = next; next = temp;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_gridA));
    CUDA_CHECK(cudaFree(d_gridB));
    
    return ms;
}

/**
 * Runs the simulation multiple times to gather statistical data on performance.
 * Performs warmup runs followed by measurement runs.
 */
void runBenchmarkWithAveraging(int width, int height, int generations, 
                                unsigned long seed, int warmup_runs, int measure_runs) {
    float* timings = (float*)malloc(measure_runs * sizeof(float));
    
    printf("Running benchmark: %d warmup + %d measurement runs\n", warmup_runs, measure_runs);

    
    // Warmup runs
    for (int i = 0; i < warmup_runs; i++) {
        runSimulation(width, height, generations, seed);
        printf("\rWarmup %d/%d complete", i+1, warmup_runs);
        fflush(stdout);
    }
    if (warmup_runs > 0) {
        printf("\n");
    }
    
    // Measurement runs
    for (int i = 0; i < measure_runs; i++) {
        timings[i] = runSimulation(width, height, generations, seed);
        printf("\rRun %d/%d: %.2f ms", i+1, measure_runs, timings[i]);
        fflush(stdout);
    }
    printf("\n\n");
    
    float mean = 0.0f, min = timings[0], max = timings[0];
    for (int i = 0; i < measure_runs; i++) {
        mean += timings[i];
        if (timings[i] < min) min = timings[i];
        if (timings[i] > max) max = timings[i];
    }
    mean /= measure_runs;
    
    float variance = 0.0f;
    for (int i = 0; i < measure_runs; i++) {
        float diff = timings[i] - mean;
        variance += diff * diff;
    }
    float std_dev = sqrt(variance / (measure_runs - 1));
    
    float* sorted = (float*)malloc(measure_runs * sizeof(float));
    memcpy(sorted, timings, measure_runs * sizeof(float));
    for (int i = 0; i < measure_runs - 1; i++) {
        for (int j = i + 1; j < measure_runs; j++) {
            if (sorted[j] < sorted[i]) {
                float temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }
    float median = (measure_runs % 2 == 0) 
        ? (sorted[measure_runs/2 - 1] + sorted[measure_runs/2]) / 2.0f
        : sorted[measure_runs/2];
    
    float cv_percent = (std_dev / mean) * 100.0f;
    double mean_throughput = (double)width * height * generations / mean / 1000.0;
    double std_throughput = mean_throughput * (std_dev / mean);
    
    printf("--- STATISTICS (n=%d) ---\n", measure_runs);
    printf("Mean time: %.2f ± %.2f ms\n", mean, std_dev);
    printf("Median time: %.2f ms\n", median);
    printf("Range: [%.2f, %.2f] ms\n", min, max);
    printf("Coefficient of Variation: %.2f%%\n", cv_percent);
    printf("Time per generation: %.4f ms\n", mean / generations);
    printf("Mean throughput: %.2f ± %.2f M cells/sec\n", mean_throughput, std_throughput);
    
    // Memory bandwidth analysis
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peakBandwidth = prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    float actualBandwidth = (width * height * 2 * sizeof(unsigned char) * generations) / (mean / 1000.0) / 1e9;
    
    printf("\n--- MEMORY BANDWIDTH ---\n");
    printf("Peak bandwidth: %.1f GB/s\n", peakBandwidth);
    printf("Achieved bandwidth: %.1f GB/s\n", actualBandwidth);
    printf("Efficiency: %.1f%%\n", actualBandwidth / peakBandwidth * 100.0);
    
    free(timings);
    free(sorted);
}

/**
 * Specialized benchmark mode that sweeps through different grid sizes
 * and outputs results to a CSV file.
 */
void runBenchmarkMode() {
    const int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int num_sizes = 8;
    const int generations = 100;
    const int warmup_runs = 2;
    const int measure_runs = 10;
    
    printf("================================================================================\n");
    printf("BENCHMARK MODE: CUDA Game of Life\n");
    printf("Configuration: %d warmup + %d measurement runs per size\n", warmup_runs, measure_runs);
    printf("================================================================================\n");
    printf("\n");
    
    FILE* csv = fopen("benchmark_cuda.csv", "w");
    fprintf(csv, "size,generations,runs,mean_time_ms,std_time_ms,median_time_ms,min_time_ms,max_time_ms,");
    fprintf(csv, "ci_95_range_ms,mean_throughput_mcells_s,std_throughput_mcells_s,cv_percent\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        printf("\n================================================================================\n");
        printf("Grid size: %dx%d\n", size, size);
        printf("================================================================================\n");
        
        float* timings = (float*)malloc(measure_runs * sizeof(float));
        
        for (int w = 0; w < warmup_runs; w++) {
            runSimulation(size, size, generations, DEFAULT_SEED);
        }
        printf("Warmup complete (%d runs)\n", warmup_runs);
        
        for (int m = 0; m < measure_runs; m++) {
            timings[m] = runSimulation(size, size, generations, DEFAULT_SEED);
            printf("\rRun %d/%d: %.2f ms", m+1, measure_runs, timings[m]);
            fflush(stdout);
        }
        printf("\n");
        
        float mean = 0.0f, min = timings[0], max = timings[0];
        for (int j = 0; j < measure_runs; j++) {
            mean += timings[j];
            if (timings[j] < min) min = timings[j];
            if (timings[j] > max) max = timings[j];
        }
        mean /= measure_runs;
        
        float variance = 0.0f;
        for (int j = 0; j < measure_runs; j++) {
            float diff = timings[j] - mean;
            variance += diff * diff;
        }
        float std_dev = sqrt(variance / (measure_runs - 1));
        
        float* sorted = (float*)malloc(measure_runs * sizeof(float));
        memcpy(sorted, timings, measure_runs * sizeof(float));
        for (int a = 0; a < measure_runs - 1; a++) {
            for (int b = a + 1; b < measure_runs; b++) {
                if (sorted[b] < sorted[a]) {
                    float temp = sorted[a];
                    sorted[a] = sorted[b];
                    sorted[b] = temp;
                }
            }
        }
        float median = (measure_runs % 2 == 0) 
            ? (sorted[measure_runs/2 - 1] + sorted[measure_runs/2]) / 2.0f
            : sorted[measure_runs/2];
        
        float cv_percent = (std_dev / mean) * 100.0f;
        double mean_throughput = (double)size * size * generations / mean / 1000.0;
        double std_throughput = mean_throughput * (std_dev / mean);
        
        printf("\nMean: %.2f ± %.2f ms (CV: %.2f%%)\n", mean, std_dev, cv_percent);
        printf("Throughput: %.2f ± %.2f M cells/s\n", mean_throughput, std_throughput);
        
        float ci_range = 1.96 * std_dev / sqrt(measure_runs);
        fprintf(csv, "%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                size, generations, measure_runs, mean, std_dev, median, min, max,
                ci_range, mean_throughput, std_throughput, cv_percent);
        
        free(timings);
        free(sorted);
    }
    
    fclose(csv);
    printf("\n\nResults saved to benchmark_cuda.csv\n");
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        runBenchmarkMode();
        return 0;
    }
    

    
    int width = (argc > 1) ? atoi(argv[1]) : 1024;
    int height = (argc > 2) ? atoi(argv[2]) : 1024;
    int generations = (argc > 3) ? atoi(argv[3]) : 100;
    int enable_averaging = (argc > 4) ? atoi(argv[4]) : 0;
    unsigned long seed = (argc > 5) ? atol(argv[5]) : DEFAULT_SEED;
    
    printf("Game of Life - CUDA Implementation\n");
    printf("Grid: %dx%d\n", width, height);
    printf("Generations: %d\n\n", generations);
    

    
    if (enable_averaging) {
        printf("--- RUNNING SIMULATION WITH AVERAGING ---\n");
        runBenchmarkWithAveraging(width, height, generations, seed, 2, 10);
    } else {
        printf("--- RUNNING SIMULATION (single run) ---\n");
        float ms = runSimulation(width, height, generations, seed);
        
        printf("\n--- RESULTS ---\n");
        printf("Total time: %.2f ms\n", ms);
        printf("Time per generation: %.4f ms\n", ms / generations);
        printf("Throughput: %.2f M cells/sec\n", 
               (double)width * height * generations / ms / 1000.0);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        float peakBandwidth = prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2 / 1e9;
        float actualBandwidth = (width * height * 2 * sizeof(unsigned char) * generations) / (ms / 1000.0) / 1e9;
        
        printf("\n--- MEMORY BANDWIDTH ---\n");
        printf("Peak bandwidth: %.1f GB/s\n", peakBandwidth);
        printf("Achieved bandwidth: %.1f GB/s\n", actualBandwidth);
        printf("Efficiency: %.1f%%\n", actualBandwidth / peakBandwidth * 100.0);
    }
    
    return 0;
}