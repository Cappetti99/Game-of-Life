# Conway's Game of Life — Multi-Implementation

Conway's Game of Life implemented in three variants: **sequential CPU**, **GPU (CUDA)**, and **interactive visualization**.

> **Performance**: CUDA reaches up to **83×** vs Python, peaking at **38.96B cells/s** on **4096×4096** grids.

## Overview

This repo showcases the same algorithm implemented with different trade-offs:

- **Sequential (Python/NumPy)** — optimized, vectorized baseline (optional SciPy acceleration)
- **Parallel (CUDA)** — high-throughput GPU kernel with memory optimizations
- **Visual (Pygame)** — interactive, real-time simulation + pattern drawing

## Project Structure

```text
Game-of-Life/
├── src/
│   ├── python/                 # Sequential Python implementation
│   │   └── game_of_life_sequential.py
│   ├── cuda/                   # Parallel CUDA implementation
│   │   └── game_of_life.cu
│   └── visual/                 # Interactive Pygame visualization
│       └── game_of_life_visual.py
├── benchmarks/                 # Benchmark results (CSV)
│   ├── benchmark_sequential.csv
│   ├── benchmark_cuda.csv
│   └── block_size_comparison.csv
├── docs/
│   └── Report_Game_of_Life.pdf
├── build/                      # Generated binaries
├── run.sh                      # Main runner
├── benchmark_block_sizes.sh    # Block-size sweep
└── README.md
```

## Quick Start

```bash
./run.sh help
./run.sh visual
./run.sh sequential
./run.sh cuda
./run.sh benchmark
./benchmark_block_sizes.sh
./run.sh clean
```

## Rules (B3/S23)

* A dead cell with **3** live neighbors becomes alive.
* A live cell survives with **2 or 3** live neighbors.
* Otherwise it dies (under/overpopulation).

## Visual Version (Pygame)

### Requirements

* Python 3.10+
* `pip install pygame`

### Run

```bash
./run.sh visual                 # default: 120x80, 10px cells
./run.sh visual 200 150 5       # width height cell_size
```

### Controls

| Key         | Action                     |
| ----------- | -------------------------- |
| **SPACE**   | Pause/Resume               |
| **R**       | Random reset               |
| **C**       | Clear                      |
| **G**       | Glider at mouse            |
| **U**       | Gosper glider gun at mouse |
| **LMB**     | Draw                       |
| **RMB**     | Erase                      |
| **UP/DOWN** | Speed +/-                  |
| **+/-**     | Zoom in/out                |
| **ESC**     | Quit                       |

## Python Sequential (NumPy / SciPy)

Vectorized implementation; optionally faster with SciPy convolution.

### Requirements

* Python 3.10+
* `pip install numpy`
* Recommended: `pip install scipy` (≈2–3× faster)

### Run

```bash
./run.sh sequential                 # default: 64x64, 100 gens
./run.sh sequential 128 128 500     # width height generations
./run.sh sequential --benchmark
```

### Notes

* Fully vectorized (no Python loops over cells)
* Toroidal boundaries (wrap-around)
* Optional `scipy.ndimage.convolve` acceleration
* Benchmark mode: multiple runs + CSV export

## CUDA Implementation

High-performance GPU kernel tuned for stencil-like workloads.

### Requirements

* NVIDIA GPU (Compute Capability 3.0+)
* CUDA Toolkit 11+
* Linux recommended (works on Windows with CUDA)

### Run

```bash
./run.sh cuda                         # default: 1024x1024, 100 gens
./run.sh cuda 2048 2048 1000 0 42     # width height gens visualize(0/1) seed
./run.sh cuda --benchmark
./benchmark_block_sizes.sh
```

### Key Optimizations

* Shared-memory tiling + halo cells
* Shared-memory padding to avoid bank conflicts
* Coalesced global memory access
* Read-only cache (`__ldg`) for boundary reads
* Double buffering (ping-pong) between generations

### Block Size (summary)

`16×16` is the best overall configuration in the included sweep (best throughput/occupancy balance).

## Benchmarks

Both CPU and CUDA benchmark grid sizes: **32 → 4096**.

```bash
./run.sh benchmark
```

Outputs:

* `benchmarks/benchmark_sequential.csv`
* `benchmarks/benchmark_cuda.csv`

### Speedup (CUDA vs Python)

| Grid      | Speedup   |
| --------- | --------- |
| 32×32     | 18.2×     |
| 64×64     | 24.3×     |
| 128×128   | 26.0×     |
| 256×256   | 22.2×     |
| 512×512   | 23.4×     |
| 1024×1024 | 22.8×     |
| 2048×2048 | 25.6×     |
| 4096×4096 | **83.0×** |

## Documentation

See **`docs/Report_Game_of_Life.pdf`** for the full technical report (CUDA kernel design, block-size study, performance analysis).

## Installation

```bash
git clone https://github.com/Cappetti99/Game-of-Life.git
cd Game-of-Life

pip install numpy scipy pygame
chmod +x run.sh benchmark_block_sizes.sh

./run.sh help
```

## Contributing

Ideas:

* Bit-packing (32 cells/word)
* Warp-level primitives (`__ballot_sync`)
* Multi-GPU domain decomposition
* Streams (overlap compute/transfer)
* More rulesets + pattern I/O
* Web visualization (WebGL)

## License

Open source, educational use.
