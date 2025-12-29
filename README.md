# Game of Life

Conway's Game of Life implementation with sequential (Python) and parallel (CUDA) versions for performance comparison.

## Rules

1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors lives on
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes alive (reproduction)

## Files

| File | Description |
|------|-------------|
| `game_of_life_sequential.py` | Sequential Python implementation using NumPy |
| `game_of_life.cu` | Parallel CUDA implementation with shared memory optimization |

## Python Sequential Implementation

### Requirements

- Python 3.10+
- NumPy

### Usage

```bash
# Run with default parameters (64x64 grid, 100 generations)
python game_of_life_sequential.py

# Custom parameters: width, height, generations, visualize (0/1)
python game_of_life_sequential.py 128 128 500 0

# Run benchmarks
python game_of_life_sequential.py --benchmark
```

### Features

- Pure Python implementation (educational, slow)
- NumPy vectorized implementation (faster, default)
- Toroidal grid wrapping (edges connect to opposite sides)
- Console visualization for small grids
- Built-in benchmarking

## CUDA Parallel Implementation

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit

### Compilation

```bash
nvcc -o game_of_life game_of_life.cu
```

### Usage

```bash
# Run with default parameters (64x64 grid, 100 generations)
./game_of_life

# Custom parameters: width, height, generations, visualize (0/1), seed
./game_of_life 1024 1024 1000 0 42

# Run benchmarks
./game_of_life --benchmark
```

### Features

- Basic kernel using global memory
- Optimized kernel using shared memory with halo cells
- Double buffering to avoid read/write conflicts
- cuRAND for GPU-side random initialization
- CUDA events for accurate timing

## Benchmarking

Both implementations include a benchmark mode that tests grid sizes: 32, 64, 128, 256, 512, and 1024.

```bash
# Python
python game_of_life_sequential.py --benchmark

# CUDA
./game_of_life --benchmark
```

Results are saved to CSV files:
- `benchmark_sequential.csv`
- `benchmark_cuda.csv`

## Performance Comparison

### Python Sequential Results (NumPy)

| Grid Size | Time/Generation | Throughput |
|-----------|-----------------|------------|
| 32x32     | 0.07 ms         | 13.8 M cells/s |
| 64x64     | 0.08 ms         | 51.2 M cells/s |
| 128x128   | 0.14 ms         | 117.8 M cells/s |
| 256x256   | 0.34 ms         | 192.3 M cells/s |
| 512x512   | 1.15 ms         | 228.3 M cells/s |
| 1024x1024 | 4.76 ms         | 220.3 M cells/s |

### CUDA Parallel Results

Run `./game_of_life --benchmark` on a machine with a GPU to generate results.

Expected speedup: 10-100x over Python for large grids, depending on GPU.
