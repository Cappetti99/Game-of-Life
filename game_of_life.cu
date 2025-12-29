/**
 * Conway's Game of Life - CUDA Parallel Implementation
 * 
 * Rules:
 * 1. Any live cell with fewer than two live neighbors dies (underpopulation)
 * 2. Any live cell with two or three live neighbors lives on
 * 3. Any live cell with more than three live neighbors dies (overpopulation)
 * 4. Any dead cell with exactly three live neighbors becomes alive (reproduction)
 *
 * Compilation: nvcc -o game_of_life game_of_life.cu
 * Usage: ./game_of_life [width] [height] [generations] [visualize]
 *        ./game_of_life --benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 16
#define DEFAULT_SEED 42  // Fixed seed for reproducibility when comparing with Python

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/**
 * Kernel to initialize the grid with random values
 */
__global__ void initGridKernel(unsigned char* grid, int width, int height, 
                                unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Initialize curand state for this thread
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        // ~30% chance of being alive initially
        grid[idx] = (curand_uniform(&state) < 0.3f) ? 1 : 0;
    }
}

/**
 * Count live neighbors for a cell at position (x, y)
 * Uses toroidal wrapping (edges connect to opposite sides)
 */
__device__ int countNeighbors(const unsigned char* grid, int x, int y, 
                               int width, int height) {
    int count = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;  // Skip the cell itself
            
            // Toroidal wrapping
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            
            count += grid[ny * width + nx];
        }
    }
    
    return count;
}

/**
 * Main Game of Life kernel - computes the next generation
 * Each thread processes one cell
 */
__global__ void gameOfLifeKernel(const unsigned char* currentGrid, 
                                  unsigned char* nextGrid,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int neighbors = countNeighbors(currentGrid, x, y, width, height);
        unsigned char currentState = currentGrid[idx];
        
        // Apply Game of Life rules
        unsigned char nextState;
        if (currentState == 1) {
            // Live cell
            nextState = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        } else {
            // Dead cell
            nextState = (neighbors == 3) ? 1 : 0;
        }
        
        nextGrid[idx] = nextState;
    }
}

/**
 * Optimized kernel using shared memory for better performance
 * Loads a tile of cells into shared memory to reduce global memory accesses
 */
__global__ void gameOfLifeKernelShared(const unsigned char* currentGrid,
                                        unsigned char* nextGrid,
                                        int width, int height) {
    // Shared memory tile with halo cells for neighbors
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;  // Offset by 1 for halo
    int ty = threadIdx.y + 1;
    
    // Load center cells
    if (x < width && y < height) {
        tile[ty][tx] = currentGrid[y * width + x];
    }
    
    // Load halo cells (edges and corners)
    // Left edge
    if (threadIdx.x == 0) {
        int nx = (x - 1 + width) % width;
        tile[ty][0] = currentGrid[y * width + nx];
    }
    // Right edge
    if (threadIdx.x == blockDim.x - 1 || x == width - 1) {
        int nx = (x + 1) % width;
        tile[ty][tx + 1] = currentGrid[y * width + nx];
    }
    // Top edge
    if (threadIdx.y == 0) {
        int ny = (y - 1 + height) % height;
        tile[0][tx] = currentGrid[ny * width + x];
    }
    // Bottom edge
    if (threadIdx.y == blockDim.y - 1 || y == height - 1) {
        int ny = (y + 1) % height;
        tile[ty + 1][tx] = currentGrid[ny * width + x];
    }
    
    // Corners
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int nx = (x - 1 + width) % width;
        int ny = (y - 1 + height) % height;
        tile[0][0] = currentGrid[ny * width + nx];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        int nx = (x + 1) % width;
        int ny = (y - 1 + height) % height;
        tile[0][tx + 1] = currentGrid[ny * width + nx];
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        int nx = (x - 1 + width) % width;
        int ny = (y + 1) % height;
        tile[ty + 1][0] = currentGrid[ny * width + nx];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        int nx = (x + 1) % width;
        int ny = (y + 1) % height;
        tile[ty + 1][tx + 1] = currentGrid[ny * width + nx];
    }
    
    // Synchronize to ensure all shared memory is loaded
    __syncthreads();
    
    if (x < width && y < height) {
        // Count neighbors from shared memory
        int neighbors = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                neighbors += tile[ty + dy][tx + dx];
            }
        }
        
        unsigned char currentState = tile[ty][tx];
        unsigned char nextState;
        
        if (currentState == 1) {
            nextState = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        } else {
            nextState = (neighbors == 3) ? 1 : 0;
        }
        
        nextGrid[y * width + x] = nextState;
    }
}

/**
 * Print the grid to the console (for small grids)
 */
void printGrid(const unsigned char* grid, int width, int height) {
    printf("\033[H");  // Move cursor to home position (for animation effect)
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%c", grid[y * width + x] ? '#' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Count total live cells in the grid
 */
int countLiveCells(const unsigned char* grid, int width, int height) {
    int count = 0;
    for (int i = 0; i < width * height; i++) {
        count += grid[i];
    }
    return count;
}

/**
 * Initialize grid with a glider pattern (for testing)
 */
void initGlider(unsigned char* grid, int width, int height, int startX, int startY) {
    // Clear grid
    memset(grid, 0, width * height);
    
    // Glider pattern
    //   #
    //     #
    // # # #
    if (startX + 2 < width && startY + 2 < height) {
        grid[startY * width + (startX + 1)] = 1;
        grid[(startY + 1) * width + (startX + 2)] = 1;
        grid[(startY + 2) * width + startX] = 1;
        grid[(startY + 2) * width + (startX + 1)] = 1;
        grid[(startY + 2) * width + (startX + 2)] = 1;
    }
}

/**
 * Initialize grid with a Gosper Glider Gun (for testing)
 */
void initGliderGun(unsigned char* grid, int width, int height) {
    memset(grid, 0, width * height);
    
    if (width < 40 || height < 20) {
        printf("Grid too small for glider gun, need at least 40x20\n");
        return;
    }
    
    int ox = 1, oy = 5;  // Offset
    
    // Left square
    grid[(oy + 0) * width + (ox + 0)] = 1;
    grid[(oy + 0) * width + (ox + 1)] = 1;
    grid[(oy + 1) * width + (ox + 0)] = 1;
    grid[(oy + 1) * width + (ox + 1)] = 1;
    
    // Left part
    grid[(oy + 0) * width + (ox + 10)] = 1;
    grid[(oy + 1) * width + (ox + 10)] = 1;
    grid[(oy + 2) * width + (ox + 10)] = 1;
    grid[(oy - 1) * width + (ox + 11)] = 1;
    grid[(oy + 3) * width + (ox + 11)] = 1;
    grid[(oy - 2) * width + (ox + 12)] = 1;
    grid[(oy + 4) * width + (ox + 12)] = 1;
    grid[(oy - 2) * width + (ox + 13)] = 1;
    grid[(oy + 4) * width + (ox + 13)] = 1;
    grid[(oy + 1) * width + (ox + 14)] = 1;
    grid[(oy - 1) * width + (ox + 15)] = 1;
    grid[(oy + 3) * width + (ox + 15)] = 1;
    grid[(oy + 0) * width + (ox + 16)] = 1;
    grid[(oy + 1) * width + (ox + 16)] = 1;
    grid[(oy + 2) * width + (ox + 16)] = 1;
    grid[(oy + 1) * width + (ox + 17)] = 1;
    
    // Right part
    grid[(oy - 2) * width + (ox + 20)] = 1;
    grid[(oy - 1) * width + (ox + 20)] = 1;
    grid[(oy + 0) * width + (ox + 20)] = 1;
    grid[(oy - 2) * width + (ox + 21)] = 1;
    grid[(oy - 1) * width + (ox + 21)] = 1;
    grid[(oy + 0) * width + (ox + 21)] = 1;
    grid[(oy - 3) * width + (ox + 22)] = 1;
    grid[(oy + 1) * width + (ox + 22)] = 1;
    grid[(oy - 4) * width + (ox + 24)] = 1;
    grid[(oy - 3) * width + (ox + 24)] = 1;
    grid[(oy + 1) * width + (ox + 24)] = 1;
    grid[(oy + 2) * width + (ox + 24)] = 1;
    
    // Right square
    grid[(oy - 2) * width + (ox + 34)] = 1;
    grid[(oy - 2) * width + (ox + 35)] = 1;
    grid[(oy - 1) * width + (ox + 34)] = 1;
    grid[(oy - 1) * width + (ox + 35)] = 1;
}

/**
 * Run a single simulation and return timing results
 */
float runSimulation(int width, int height, int generations, int visualize, 
                    unsigned long seed, int useSharedMemory) {
    size_t gridSize = width * height * sizeof(unsigned char);
    
    // Allocate host memory
    unsigned char* h_grid = (unsigned char*)malloc(gridSize);
    if (!h_grid) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }
    
    // Allocate device memory (double buffering)
    unsigned char *d_gridA, *d_gridB;
    CUDA_CHECK(cudaMalloc(&d_gridA, gridSize));
    CUDA_CHECK(cudaMalloc(&d_gridB, gridSize));
    
    // Set up grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Initialize grid on GPU with random values
    initGridKernel<<<gridDim, blockDim>>>(d_gridA, width, height, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy initial state to host for display/verification
    CUDA_CHECK(cudaMemcpy(h_grid, d_gridA, gridSize, cudaMemcpyDeviceToHost));
    
    if (!visualize) {
        printf("Initial live cells: %d\n", countLiveCells(h_grid, width, height));
    }
    
    if (visualize && width <= 80 && height <= 40) {
        printf("\033[2J");  // Clear screen
        printGrid(h_grid, width, height);
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Main simulation loop
    CUDA_CHECK(cudaEventRecord(start));
    
    unsigned char *currentGrid = d_gridA;
    unsigned char *nextGrid = d_gridB;
    
    for (int gen = 0; gen < generations; gen++) {
        if (useSharedMemory) {
            gameOfLifeKernelShared<<<gridDim, blockDim>>>(currentGrid, nextGrid, 
                                                          width, height);
        } else {
            gameOfLifeKernel<<<gridDim, blockDim>>>(currentGrid, nextGrid, 
                                                     width, height);
        }
        
        // Swap buffers
        unsigned char* temp = currentGrid;
        currentGrid = nextGrid;
        nextGrid = temp;
        
        // Optional: visualize each generation (slow, for debugging)
        if (visualize && width <= 80 && height <= 40) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_grid, currentGrid, gridSize, 
                                  cudaMemcpyDeviceToHost));
            printGrid(h_grid, width, height);
            printf("Generation: %d, Live cells: %d\n", gen + 1, 
                   countLiveCells(h_grid, width, height));
            
            // Small delay for visualization
            struct timespec ts = {0, 100000000};  // 100ms
            nanosleep(&ts, NULL);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy final state to host
    CUDA_CHECK(cudaMemcpy(h_grid, currentGrid, gridSize, cudaMemcpyDeviceToHost));
    
    if (!visualize) {
        printf("Final live cells: %d\n", countLiveCells(h_grid, width, height));
    }
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_gridA));
    CUDA_CHECK(cudaFree(d_gridB));
    free(h_grid);
    
    return milliseconds;
}

/**
 * Run benchmarks for different grid sizes
 * @param generations Number of generations to run
 * @param verbose If true, print detailed output for each size
 */
void benchmark(int generations, int verbose) {
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("============================================================\n");
    printf("BENCHMARK: CUDA Parallel Game of Life\n");
    printf("============================================================\n\n");
    
    float results[8];  // Store results for summary
    
    for (int i = 0; i < numSizes; i++) {
        int size = sizes[i];
        
        if (verbose) {
            printf("\n--- Grid size: %dx%d ---\n", size, size);
            printf("Game of Life CUDA Implementation\n");
            printf("Grid size: %d x %d\n", size, size);
            printf("Generations: %d\n", generations);
            printf("Using shared memory kernel\n\n");
        }
        
        float ms = runSimulation(size, size, generations, 0, DEFAULT_SEED, 1);
        results[i] = ms;
        
        if (verbose) {
            printf("\nSimulation complete!\n");
            printf("Total time: %.2f ms\n", ms);
            printf("Time per generation: %.4f ms\n", ms / generations);
            printf("Cells processed per second: %.2f million\n\n", 
                   (double)size * size * generations / ms / 1000.0);
        } else {
            printf("Size %5dx%-5d done: %10.2f ms\n", size, size, ms);
        }
    }
    
    // Summary table
    printf("\n============================================================\n");
    printf("SUMMARY\n");
    printf("============================================================\n");
    printf("%10s | %12s | %14s | %12s\n", "Size", "Total (ms)", "Per Gen (ms)", "M cells/s");
    printf("------------------------------------------------------------\n");
    for (int i = 0; i < numSizes; i++) {
        int size = sizes[i];
        float ms = results[i];
        printf("%10d | %12.2f | %14.4f | %12.2f\n", 
               size, ms, ms / generations,
               (double)size * size * generations / ms / 1000.0);
    }
    printf("============================================================\n");
    
    // Save results to CSV for easy comparison
    FILE* f = fopen("benchmark_cuda.csv", "w");
    if (f) {
        fprintf(f, "size,generations,total_time_ms,time_per_generation_ms,cells_per_second_million\n");
        for (int i = 0; i < numSizes; i++) {
            int size = sizes[i];
            float ms = results[i];
            fprintf(f, "%d,%d,%.4f,%.6f,%.4f\n",
                    size, generations, ms, ms / generations,
                    (double)size * size * generations / ms / 1000.0);
        }
        fclose(f);
        printf("\nResults saved to benchmark_cuda.csv\n");
    }
}

int main(int argc, char** argv) {
    // Check for benchmark mode
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        int generations = 100;
        int verbose = 1;  // Set to 0 for quiet mode (only summary)
        if (argc > 2) generations = atoi(argv[2]);
        benchmark(generations, verbose);
        return EXIT_SUCCESS;
    }
    
    // Default parameters
    int width = 64;
    int height = 64;
    int generations = 100;
    int useSharedMemory = 1;  // Use optimized shared memory kernel by default
    int visualize = 0;        // Set to 1 for console visualization (small grids only)
    unsigned long seed = DEFAULT_SEED;  // Fixed seed for reproducibility
    
    // Parse command line arguments
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) generations = atoi(argv[3]);
    if (argc > 4) visualize = atoi(argv[4]);
    if (argc > 5) seed = atol(argv[5]);
    
    printf("Game of Life CUDA Implementation\n");
    printf("Grid size: %d x %d\n", width, height);
    printf("Generations: %d\n", generations);
    printf("Using %s kernel\n", useSharedMemory ? "shared memory" : "global memory");
    printf("Seed: %lu\n", seed);
    printf("\n");
    
    float milliseconds = runSimulation(width, height, generations, visualize, 
                                        seed, useSharedMemory);
    
    // Print results
    printf("\nSimulation complete!\n");
    printf("Total time: %.2f ms\n", milliseconds);
    printf("Time per generation: %.4f ms\n", milliseconds / generations);
    printf("Cells processed per second: %.2f million\n", 
           (double)width * height * generations / milliseconds / 1000.0);
    
    return EXIT_SUCCESS;
}
