/**
 * Game of Life - Block Size Performance Tester
 * 
 * Compila con diversi BLOCK_SIZE e confronta performance
 * 
 * Compilazione:
 * nvcc -o test_bs4  -DBLOCK_SIZE=4  game_of_life_test.cu
 * nvcc -o test_bs8  -DBLOCK_SIZE=8  game_of_life_test.cu
 * nvcc -o test_bs16 -DBLOCK_SIZE=16 game_of_life_test.cu
 * nvcc -o test_bs32 -DBLOCK_SIZE=32 game_of_life_test.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// BLOCK_SIZE viene passato da linea di comando durante compilazione
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#define DEFAULT_SEED 42

// Calcola dimensione shared memory a compile-time
#define TILE_SIZE (BLOCK_SIZE + 2)
#define SHARED_MEM_SIZE (TILE_SIZE * TILE_SIZE)

// Macro per stampare info sulla configurazione
#define PRINT_CONFIG() \
    printf("=================================================\n"); \
    printf("Configuration:\n"); \
    printf("  BLOCK_SIZE: %dx%d = %d threads/block\n", \
           BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * BLOCK_SIZE); \
    printf("  TILE_SIZE: %dx%d (include halo)\n", TILE_SIZE, TILE_SIZE); \
    printf("  Shared Memory: %d bytes/block\n", SHARED_MEM_SIZE); \
    printf("=================================================\n\n");

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// KERNEL con BLOCK_SIZE configurabile
// ============================================================================

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
    // Shared memory size si adatta automaticamente a BLOCK_SIZE
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Carica cella centrale
    if (x < width && y < height) {
        tile[ty][tx] = currentGrid[y * width + x];
    }
    
    // Carica halo cells con wrapping toroidale
    if (threadIdx.x == 0) {
        int nx = (x - 1 + width) % width;
        if (y < height) tile[ty][0] = currentGrid[y * width + nx];
    }
    if (threadIdx.x == blockDim.x - 1 || x == width - 1) {
        int nx = (x + 1) % width;
        if (y < height) tile[ty][tx + 1] = currentGrid[y * width + nx];
    }
    if (threadIdx.y == 0) {
        int ny = (y - 1 + height) % height;
        if (x < width) tile[0][tx] = currentGrid[ny * width + x];
    }
    if (threadIdx.y == blockDim.y - 1 || y == height - 1) {
        int ny = (y + 1) % height;
        if (x < width) tile[ty + 1][tx] = currentGrid[ny * width + x];
    }
    
    // Carica corners
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
    
    // Conta vicini e applica regole
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

// ============================================================================
// ANALISI OCCUPANCY
// ============================================================================

void analyzeOccupancy(int width, int height) {
    printf("\n--- OCCUPANCY ANALYSIS ---\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("\n");
    
    // Calcola grid configuration
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
    int totalBlocks = gridDim.x * gridDim.y;
    int totalThreads = totalBlocks * threadsPerBlock;
    
    printf("Grid Configuration:\n");
    printf("  Block Dim: %dx%d = %d threads/block\n", 
           BLOCK_SIZE, BLOCK_SIZE, threadsPerBlock);
    printf("  Grid Dim: %dx%d = %d blocks\n", 
           gridDim.x, gridDim.y, totalBlocks);
    printf("  Total threads: %d\n", totalThreads);
    printf("\n");
    
    // Calcola theoretical occupancy
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / threadsPerBlock;
    
    // Limiti di shared memory
    int sharedMemPerBlock = SHARED_MEM_SIZE;
    int maxBlocksByShmem = prop.sharedMemPerMultiprocessor / sharedMemPerBlock;
    
    // Limiti di blocchi
    int maxBlocksByLimit = prop.maxBlocksPerMultiProcessor;
    
    int actualBlocksPerSM = maxBlocksPerSM;
    if (maxBlocksByShmem < actualBlocksPerSM) actualBlocksPerSM = maxBlocksByShmem;
    if (maxBlocksByLimit < actualBlocksPerSM) actualBlocksPerSM = maxBlocksByLimit;
    
    float theoreticalOccupancy = (float)(actualBlocksPerSM * threadsPerBlock) / 
                                  prop.maxThreadsPerMultiProcessor * 100.0f;
    
    printf("Occupancy Analysis:\n");
    printf("  Max blocks/SM (by threads): %d\n", maxBlocksPerSM);
    printf("  Max blocks/SM (by shmem): %d\n", maxBlocksByShmem);
    printf("  Max blocks/SM (by limit): %d\n", maxBlocksByLimit);
    printf("  → Actual blocks/SM: %d\n", actualBlocksPerSM);
    printf("  → Theoretical Occupancy: %.1f%%\n", theoreticalOccupancy);
    printf("\n");
    
    // Warp analysis
    int warpsPerBlock = (threadsPerBlock + 31) / 32;
    printf("Warp Efficiency:\n");
    printf("  Warps per block: %d\n", warpsPerBlock);
    printf("  Last warp utilization: %d/32 threads\n", 
           threadsPerBlock - (warpsPerBlock - 1) * 32);
    if (threadsPerBlock % 32 == 0) {
        printf("  ✓ Perfect warp alignment!\n");
    } else {
        printf("  ⚠ Partial warp (inefficient)\n");
    }
    printf("\n");
}

// ============================================================================
// BENCHMARK
// ============================================================================

float runSimulation(int width, int height, int generations, unsigned long seed) {
    size_t gridSize = width * height * sizeof(unsigned char);
    
    unsigned char *d_gridA, *d_gridB;
    CUDA_CHECK(cudaMalloc(&d_gridA, gridSize));
    CUDA_CHECK(cudaMalloc(&d_gridB, gridSize));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Inizializzazione
    initGridKernel<<<gridDim, blockDim>>>(d_gridA, width, height, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
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
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_gridA));
    CUDA_CHECK(cudaFree(d_gridB));
    
    return ms;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    PRINT_CONFIG();
    
    int width = (argc > 1) ? atoi(argv[1]) : 1024;
    int height = (argc > 2) ? atoi(argv[2]) : 1024;
    int generations = (argc > 3) ? atoi(argv[3]) : 100;
    
    printf("Game of Life - Block Size Test\n");
    printf("Grid: %dx%d\n", width, height);
    printf("Generations: %d\n\n", generations);
    
    // Analizza occupancy
    analyzeOccupancy(width, height);
    
    // Esegui simulazione
    printf("--- RUNNING SIMULATION ---\n");
    float ms = runSimulation(width, height, generations, DEFAULT_SEED);
    
    printf("\n--- RESULTS ---\n");
    printf("Total time: %.2f ms\n", ms);
    printf("Time per generation: %.4f ms\n", ms / generations);
    printf("Throughput: %.2f M cells/sec\n", 
           (double)width * height * generations / ms / 1000.0);
    
    // Calcola efficienza rispetto al teorico
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peakBandwidth = prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2 / 1e9;
    float actualBandwidth = (width * height * 2 * sizeof(unsigned char) * generations) / (ms / 1000.0) / 1e9;
    
    printf("\n--- MEMORY BANDWIDTH ---\n");
    printf("Peak bandwidth: %.1f GB/s\n", peakBandwidth);
    printf("Achieved bandwidth: %.1f GB/s\n", actualBandwidth);
    printf("Efficiency: %.1f%%\n", actualBandwidth / peakBandwidth * 100.0);
    
    return 0;
}

/* ============================================================================
 * ISTRUZIONI D'USO:
 * 
 * 1. Compila con diversi BLOCK_SIZE:
 *    nvcc -o test_bs4  -DBLOCK_SIZE=4  game_of_life_test.cu
 *    nvcc -o test_bs8  -DBLOCK_SIZE=8  game_of_life_test.cu
 *    nvcc -o test_bs16 -DBLOCK_SIZE=16 game_of_life_test.cu
 *    nvcc -o test_bs32 -DBLOCK_SIZE=32 game_of_life_test.cu
 * 
 * 2. Esegui tutti:
 *    ./test_bs4  1024 1024 100
 *    ./test_bs8  1024 1024 100
 *    ./test_bs16 1024 1024 100
 *    ./test_bs32 1024 1024 100
 * 
 * 3. Confronta risultati!
 * 
 * COSA ASPETTARSI:
 * - BS=4:  Bassa occupancy, molti blocchi, overhead alto
 * - BS=8:  Buon bilanciamento, alta occupancy
 * - BS=16: OTTIMO (default), migliori performance
 * - BS=32: Troppi thread/block, occupancy ridotta
 * 
 * ============================================================================
 */