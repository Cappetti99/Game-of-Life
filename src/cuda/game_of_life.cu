/**
 * Conway's Game of Life - CUDA Parallel Implementation
 * 
 * Compilation: nvcc -o game_of_life game_of_life.cu
 * Usage: ./game_of_life [width] [height] [generations] [visualize] [seed]
 *        ./game_of_life --benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif
#define DEFAULT_SEED 42

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void initGridKernel(unsigned char* grid, int width, int height, 
                                unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        curandState state;
        curand_init(seed, idx, 0, &state);
        grid[idx] = (curand_uniform(&state) < 0.3f) ? 1 : 0;
    }
}

__global__ void gameOfLifeKernel(const unsigned char* currentGrid,
                                  unsigned char* nextGrid,
                                  int width, int height) {
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load center cells
    if (x < width && y < height) {
        tile[ty][tx] = currentGrid[y * width + x];
    }
    
    // Load halo cells
    if (threadIdx.x == 0) {
        int nx = (x - 1 + width) % width;
        tile[ty][0] = currentGrid[y * width + nx];
    }
    if (threadIdx.x == blockDim.x - 1 || x == width - 1) {
        int nx = (x + 1) % width;
        tile[ty][tx + 1] = currentGrid[y * width + nx];
    }
    if (threadIdx.y == 0) {
        int ny = (y - 1 + height) % height;
        tile[0][tx] = currentGrid[ny * width + x];
    }
    if (threadIdx.y == blockDim.y - 1 || y == height - 1) {
        int ny = (y + 1) % height;
        tile[ty + 1][tx] = currentGrid[ny * width + x];
    }
    
    // Load corners
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
    
    __syncthreads();
    
    if (x < width && y < height) {
        int neighbors = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                neighbors += tile[ty + dy][tx + dx];
            }
        }
        
        unsigned char current = tile[ty][tx];
        nextGrid[y * width + x] = (current == 1) 
            ? ((neighbors == 2 || neighbors == 3) ? 1 : 0)
            : ((neighbors == 3) ? 1 : 0);
    }
}

int countLiveCells(const unsigned char* grid, int size) {
    int count = 0;
    for (int i = 0; i < size; i++) count += grid[i];
    return count;
}

float runSimulation(int width, int height, int generations, unsigned long seed) {
    size_t gridSize = width * height * sizeof(unsigned char);
    
    unsigned char* h_grid = (unsigned char*)malloc(gridSize);
    unsigned char *d_gridA, *d_gridB;
    CUDA_CHECK(cudaMalloc(&d_gridA, gridSize));
    CUDA_CHECK(cudaMalloc(&d_gridB, gridSize));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    initGridKernel<<<gridDim, blockDim>>>(d_gridA, width, height, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_grid, d_gridA, gridSize, cudaMemcpyDeviceToHost));
    printf("Initial live cells: %d\n", countLiveCells(h_grid, width * height));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    unsigned char *current = d_gridA, *next = d_gridB;
    for (int gen = 0; gen < generations; gen++) {
        gameOfLifeKernel<<<gridDim, blockDim>>>(current, next, width, height);
        unsigned char* temp = current; current = next; next = temp;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_grid, current, gridSize, cudaMemcpyDeviceToHost));
    printf("Final live cells: %d\n", countLiveCells(h_grid, width * height));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_gridA));
    CUDA_CHECK(cudaFree(d_gridB));
    free(h_grid);
    
    return ms;
}

void benchmark(int generations) {
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    float results[8];
    
    printf("============================================================\n");
    printf("BENCHMARK: CUDA Parallel Game of Life\n");
    printf("============================================================\n\n");
    
    for (int i = 0; i < numSizes; i++) {
        int size = sizes[i];
        printf("\n--- Grid size: %dx%d ---\n", size, size);
        printf("Game of Life CUDA Implementation\n");
        printf("Grid size: %d x %d\n", size, size);
        printf("Generations: %d\n", generations);
        printf("Using shared memory kernel\n\n");
        
        float ms = runSimulation(size, size, generations, DEFAULT_SEED);
        results[i] = ms;
        
        printf("\nSimulation complete!\n");
        printf("Total time: %.2f ms\n", ms);
        printf("Time per generation: %.4f ms\n", ms / generations);
        printf("Cells processed per second: %.2f million\n\n", 
               (double)size * size * generations / ms / 1000.0);
    }
    
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
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        int generations = (argc > 2) ? atoi(argv[2]) : 100;
        benchmark(generations);
        return 0;
    }
    
    int width = (argc > 1) ? atoi(argv[1]) : 64;
    int height = (argc > 2) ? atoi(argv[2]) : 64;
    int generations = (argc > 3) ? atoi(argv[3]) : 100;
    unsigned long seed = (argc > 4) ? atol(argv[4]) : DEFAULT_SEED;
    
    printf("Game of Life CUDA Implementation\n");
    printf("Grid size: %d x %d\n", width, height);
    printf("Generations: %d\n", generations);
    printf("Using shared memory kernel\n\n");
    
    float ms = runSimulation(width, height, generations, seed);
    
    printf("\nSimulation complete!\n");
    printf("Total time: %.2f ms\n", ms);
    printf("Time per generation: %.4f ms\n", ms / generations);
    printf("Cells processed per second: %.2f million\n", 
           (double)width * height * generations / ms / 1000.0);
    
    return 0;
}
