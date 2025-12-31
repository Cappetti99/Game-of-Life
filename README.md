# Game of Life

Conway's Game of Life implementation with three versions:
- **Sequential** (Python/NumPy) - For learning and prototyping
- **Parallel** (CUDA) - High-performance GPU computation
- **Visual** (Pygame) - Interactive real-time visualization

## Project Structure

```
Game-of-Life/
├── src/
│   ├── python/                 # Sequential Python implementation
│   │   └── game_of_life_sequential.py
│   ├── cuda/                   # Parallel CUDA implementation
│   │   └── game_of_life.cu
│   └── visual/                 # Interactive Pygame visualization
│       └── game_of_life_visual.py
├── benchmarks/                 # Benchmark results (CSV)
├── docs/                       # Documentation
│   └── IMPLEMENTATION.md
├── build/                      # Compiled binaries (generated)
├── run.sh                      # Main runner script
└── README.md
```

## Quick Start

```bash
# Show all options
./run.sh help

# Run visual version (interactive)
./run.sh visual

# Run sequential version
./run.sh sequential

# Run CUDA version
./run.sh cuda

# Run all benchmarks
./run.sh benchmark

# Clean build artifacts
./run.sh clean
```

## Rules

1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors lives on
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes alive (reproduction)

## Visual Version (Pygame)

Interactive visualization with drawing capabilities.

### Requirements

- Python 3.10+
- Pygame (`pip install pygame`)

### Usage

```bash
# Default (120x80 grid, 10px cells)
./run.sh visual

# Custom: width height cell_size
./run.sh visual 200 150 5
```

### Controls

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume simulation |
| **R** | Reset with random grid |
| **C** | Clear grid |
| **G** | Add glider at mouse position |
| **U** | Add glider gun at mouse position |
| **LEFT CLICK** | Draw cells |
| **RIGHT CLICK** | Erase cells |
| **UP/DOWN** | Increase/Decrease speed |
| **+/-** | Zoom in/out |
| **ESC** | Quit |

## Python Sequential Implementation

### Requirements

- Python 3.10+
- NumPy

### Usage

```bash
# Default (64x64 grid, 100 generations)
./run.sh sequential

# Custom: width height generations visualize(0/1)
./run.sh sequential 128 128 500 0

# Benchmark mode
./run.sh sequential --benchmark
```

### Features

- Pure Python implementation (educational)
- NumPy vectorized implementation (faster, default)
- Toroidal grid wrapping
- Console visualization for small grids
- Built-in benchmarking

## CUDA Parallel Implementation

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc)

### Usage

```bash
# Compile and run with defaults
./run.sh cuda

# Custom: width height generations visualize(0/1) seed
./run.sh cuda 1024 1024 1000 0 42

# Benchmark mode
./run.sh cuda --benchmark
```

### Features

- Basic kernel using global memory
- Optimized kernel using shared memory with halo cells
- Double buffering to avoid read/write conflicts
- cuRAND for GPU-side random initialization
- CUDA events for accurate timing

## Benchmarking

Both computational implementations include benchmark modes testing grid sizes: 32, 64, 128, 256, 512, 1024, 2048, and 4096.

```bash
# Run all benchmarks
./run.sh benchmark
```

Results are saved to:
- `benchmarks/benchmark_sequential.csv`
- `benchmarks/benchmark_cuda.csv`

## Performance Results

### Python Sequential (NumPy)

| Grid Size | Total Time | Time/Gen | Throughput |
|-----------|------------|----------|------------|
| 32x32 | 8.46 ms | 0.085 ms | 12 M cells/s |
| 64x64 | 7.95 ms | 0.080 ms | 52 M cells/s |
| 128x128 | 8.83 ms | 0.088 ms | 186 M cells/s |
| 256x256 | 10.52 ms | 0.105 ms | 623 M cells/s |
| 512x512 | 24.71 ms | 0.247 ms | 1,061 M cells/s |
| 1024x1024 | 71.07 ms | 0.711 ms | 1,475 M cells/s |
| 2048x2048 | 278.38 ms | 2.784 ms | 1,507 M cells/s |
| 4096x4096 | 3,920.58 ms | 39.206 ms | 428 M cells/s |

### CUDA Parallel (Shared Memory)

| Grid Size | Total Time | Time/Gen | Throughput |
|-----------|------------|----------|------------|
| 32x32 | 0.21 ms | 0.002 ms | 480 M cells/s |
| 64x64 | 0.23 ms | 0.002 ms | 1,793 M cells/s |
| 128x128 | 0.24 ms | 0.002 ms | 6,750 M cells/s |
| 256x256 | 0.38 ms | 0.004 ms | 17,418 M cells/s |
| 512x512 | 0.89 ms | 0.009 ms | 29,305 M cells/s |
| 1024x1024 | 3.01 ms | 0.030 ms | 34,873 M cells/s |
| 2048x2048 | 13.04 ms | 0.130 ms | 32,159 M cells/s |
| 4096x4096 | 44.77 ms | 0.448 ms | 37,478 M cells/s |

### Speedup (CUDA vs Python)

| Grid Size | Python | CUDA | Speedup |
|-----------|--------|------|---------|
| 32x32 | 8.46 ms | 0.21 ms | **40x** |
| 64x64 | 7.95 ms | 0.23 ms | **35x** |
| 128x128 | 8.83 ms | 0.24 ms | **37x** |
| 256x256 | 10.52 ms | 0.38 ms | **28x** |
| 512x512 | 24.71 ms | 0.89 ms | **28x** |
| 1024x1024 | 71.07 ms | 3.01 ms | **24x** |
| 2048x2048 | 278.38 ms | 13.04 ms | **21x** |
| 4096x4096 | 3,920.58 ms | 44.77 ms | **88x** |

> Peak throughput: **37.5 billion cells/second** (CUDA, 4096x4096)

## Documentation

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for detailed technical documentation including:
- Algorithm explanations
- CUDA kernel architecture
- Shared memory optimization
- Memory layout diagrams
