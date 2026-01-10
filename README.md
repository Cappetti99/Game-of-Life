# Conway's Game of Life - Multi-Implementation Project

A comprehensive implementation of Conway's Game of Life featuring three distinct versions optimized for different use cases: sequential CPU computation, massively parallel GPU acceleration, and interactive real-time visualization.

## Overview

Conway's Game of Life is a cellular automaton devised by mathematician John Conway in 1970. This project demonstrates the power of parallel computing by implementing the same algorithm in multiple ways:

- **Sequential** (Python/NumPy) - Educational implementation for learning and prototyping
- **Parallel** (CUDA) - High-performance GPU computation achieving up to 88x speedup
- **Visual** (Pygame) - Interactive real-time visualization with drawing capabilities

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

## Game Rules

The simulation follows four simple rules that create complex emergent behavior:

1. **Underpopulation**: Any live cell with fewer than two live neighbors dies
2. **Survival**: Any live cell with two or three live neighbors lives on to the next generation
3. **Overpopulation**: Any live cell with more than three live neighbors dies
4. **Reproduction**: Any dead cell with exactly three live neighbors becomes alive

These simple rules lead to fascinating patterns including oscillators, spaceships, and even Turing-complete constructions.

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

The CUDA implementation demonstrates significant performance gains over the sequential Python version, with speedup increasing for larger grid sizes.

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
> 
> The CUDA implementation shows consistent performance advantages, with speedup ranging from 21x to 88x depending on grid size. The largest grids benefit most from GPU parallelization.

## Key Features

### All Implementations
- Toroidal grid topology (edges wrap around)
- Reproducible random initialization with seed support
- Comprehensive benchmarking capabilities
- CSV export of benchmark results

### Sequential Implementation
- Pure Python and NumPy vectorized implementations
- Educational code structure for learning
- Efficient for small to medium grids
- No special hardware requirements

### CUDA Implementation
- Optimized shared memory kernel with halo cells
- Double buffering to eliminate read/write conflicts
- cuRAND for GPU-side random initialization
- CUDA events for accurate timing
- Scales efficiently to very large grids (4096x4096+)

### Visual Implementation
- Real-time interactive simulation
- Mouse-based cell drawing and erasing
- Adjustable simulation speed (1-60 FPS)
- Pre-built patterns (glider, Gosper glider gun)
- Generation counter and live cell statistics

## Documentation

- [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) - Detailed technical documentation including:
  - Algorithm explanations and pseudocode
  - CUDA kernel architecture and optimization strategies
  - Shared memory layout with halo cell handling
  - Memory management and double buffering
  - Performance analysis and bottleneck discussion

## System Requirements

### Sequential Python Implementation
- Python 3.10 or higher
- NumPy library (`pip install numpy`)
- Works on any system (Windows, Linux, macOS)

### CUDA Parallel Implementation
- NVIDIA GPU with CUDA support (Compute Capability 3.0+)
- CUDA Toolkit 11.0 or higher (includes nvcc compiler)
- Linux recommended (also works on Windows with CUDA installed)

### Visual Implementation
- Python 3.10 or higher
- Pygame library (`pip install pygame`)
- Works on any system with graphics support

## Installation

```bash
# Clone the repository
git clone https://github.com/Cappetti99/Game-of-Life.git
cd Game-of-Life

# Install Python dependencies
pip install numpy pygame

# Make the runner script executable
chmod +x run.sh

# Test the installation
./run.sh help
```

## Project Structure Details

The project is organized for clarity and separation of concerns:

- `src/python/` - Sequential CPU implementation using NumPy
- `src/cuda/` - Parallel GPU implementation using CUDA C++
- `src/visual/` - Interactive visualization using Pygame
- `benchmarks/` - Performance measurement results (generated)
- `docs/` - Detailed technical documentation
- `build/` - Compiled CUDA binaries (generated)
- `run.sh` - Unified interface for all implementations

## Contributing

Contributions are welcome! Areas for improvement:
- Additional optimization techniques (texture memory, warp-level primitives)
- Support for different cellular automaton rules
- Web-based visualization using WebGL
- Multi-GPU support for extremely large grids
- Pattern recognition and analysis tools

## License

This project is open source and available for educational purposes.

## References

- [Conway's Game of Life - Wikipedia](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [LifeWiki - Pattern Database](https://conwaylife.com/wiki/)

## Documentation

## Author

Created as a demonstration of parallel computing techniques and GPU acceleration for cellular automaton simulations.

---

For detailed implementation information, see [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).
