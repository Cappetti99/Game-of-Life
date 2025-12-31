# Game of Life - Implementation Overview

## Project Structure

```
Game-of-Life/
├── src/
│   ├── python/
│   │   └── game_of_life_sequential.py  # Python/NumPy sequential implementation
│   ├── cuda/
│   │   └── game_of_life.cu             # CUDA parallel implementation
│   └── visual/
│       └── game_of_life_visual.py      # Interactive Pygame visualization
├── benchmarks/                          # Benchmark results (CSV)
│   ├── benchmark_sequential.csv
│   └── benchmark_cuda.csv
├── docs/
│   └── IMPLEMENTATION.md               # This file - technical details
├── build/                              # Compiled binaries (generated)
├── run.sh                              # Main runner script
└── README.md                           # Project documentation
```

---

## Conway's Game of Life Rules

| Rule | Description |
|------|-------------|
| **Underpopulation** | Live cell with < 2 neighbors dies |
| **Survival** | Live cell with 2-3 neighbors lives |
| **Overpopulation** | Live cell with > 3 neighbors dies |
| **Reproduction** | Dead cell with exactly 3 neighbors becomes alive |

---

## Python Sequential Implementation

### File: `src/python/game_of_life_sequential.py`

### Core Functions

#### 1. Grid Initialization
```python
def init_random(width, height, density=0.3, seed=None):
    """Initialize grid with ~30% live cells randomly distributed"""
    return (np.random.random((height, width)) < density).astype(np.uint8)
```

#### 2. Neighbor Counting (Pure Python)
```python
def count_neighbors(grid, x, y, width, height):
    """Count live neighbors using toroidal wrapping (edges connect)"""
    count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue  # Skip self
            nx = (x + dx) % width   # Wrap horizontally
            ny = (y + dy) % height  # Wrap vertically
            count += grid[ny, nx]
    return count
```

#### 3. Game Step - NumPy Vectorized (Default)
```python
def game_of_life_step_numpy(current_grid):
    """Vectorized implementation using np.roll for neighbor counting"""
    neighbors = np.zeros_like(current_grid)
    
    # Sum all 8 neighbors using roll operations
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            neighbors += np.roll(np.roll(current_grid, dy, axis=0), dx, axis=1)
    
    # Apply rules vectorized
    birth = (current_grid == 0) & (neighbors == 3)
    survive = (current_grid == 1) & ((neighbors == 2) | (neighbors == 3))
    
    return (birth | survive).astype(np.uint8)
```

### Algorithm Flow
```
┌─────────────────────────────────────────────────────────────┐
│                    SEQUENTIAL ALGORITHM                      │
├─────────────────────────────────────────────────────────────┤
│  1. Initialize grid with random values (~30% alive)          │
│                           ↓                                  │
│  2. For each generation:                                     │
│     ┌──────────────────────────────────────────────────┐    │
│     │  a. Create neighbor count matrix using np.roll   │    │
│     │  b. Apply birth rule: dead + 3 neighbors → alive │    │
│     │  c. Apply survival: alive + 2-3 neighbors → alive│    │
│     │  d. Combine results into next generation         │    │
│     └──────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  3. Return final grid state                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## CUDA Parallel Implementation

### File: `src/cuda/game_of_life.cu`

### Key Constants
```c
#define BLOCK_SIZE 16       // 16x16 threads per block
#define DEFAULT_SEED 42     // Reproducible random initialization
```

### Core Components

#### 1. Grid Initialization Kernel
```c
__global__ void initGridKernel(unsigned char* grid, int width, int height, 
                                unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        curandState state;
        curand_init(seed, idx, 0, &state);
        grid[idx] = (curand_uniform(&state) < 0.3f) ? 1 : 0;  // 30% alive
    }
}
```

#### 2. Basic Kernel (Global Memory)
```c
__global__ void gameOfLifeKernel(const unsigned char* currentGrid, 
                                  unsigned char* nextGrid,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int neighbors = countNeighbors(currentGrid, x, y, width, height);
        unsigned char currentState = currentGrid[y * width + x];
        
        // Apply Game of Life rules
        if (currentState == 1) {
            nextGrid[y * width + x] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        } else {
            nextGrid[y * width + x] = (neighbors == 3) ? 1 : 0;
        }
    }
}
```

#### 3. Optimized Kernel (Shared Memory with Halo)
```c
__global__ void gameOfLifeKernelShared(const unsigned char* currentGrid,
                                        unsigned char* nextGrid,
                                        int width, int height) {
    // Shared memory tile with halo cells (+2 for neighbor access)
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    // Load center cells + halo (edges and corners)
    // ... loading logic ...
    
    __syncthreads();  // Ensure all data loaded
    
    // Count neighbors from shared memory (fast!)
    // Apply rules and write result
}
```

### Memory Layout & Double Buffering
```
┌─────────────────────────────────────────────────────────────┐
│                    DOUBLE BUFFERING                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    Generation N           Kernel          Generation N+1     │
│  ┌─────────────┐        ┌───────┐       ┌─────────────┐     │
│  │  d_gridA    │───────→│ CUDA  │──────→│  d_gridB    │     │
│  │  (current)  │        │Kernel │       │  (next)     │     │
│  └─────────────┘        └───────┘       └─────────────┘     │
│                                                              │
│    After kernel:  swap(d_gridA, d_gridB)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Shared Memory Tile Layout
```
┌─────────────────────────────────────────────────────────────┐
│              SHARED MEMORY TILE (18x18 for 16x16 block)      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌─┬─────────────────────────────────────────────────┬─┐  │
│    │C│              TOP HALO ROW                       │C│  │
│    ├─┼─────────────────────────────────────────────────┼─┤  │
│    │ │                                                 │ │  │
│    │L│                                                 │R│  │
│    │E│           CENTER CELLS (16x16)                  │I│  │
│    │F│         Threads process these cells             │G│  │
│    │T│                                                 │H│  │
│    │ │                                                 │T│  │
│    │H│                                                 │ │  │
│    │A│                                                 │H│  │
│    │L│                                                 │A│  │
│    │O│                                                 │L│  │
│    │ │                                                 │O│  │
│    ├─┼─────────────────────────────────────────────────┼─┤  │
│    │C│             BOTTOM HALO ROW                     │C│  │
│    └─┴─────────────────────────────────────────────────┴─┘  │
│                                                              │
│    C = Corner cells (need special handling)                  │
│    Halo cells loaded from neighboring blocks or wrapped      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### CUDA Execution Model
```
┌─────────────────────────────────────────────────────────────┐
│                  CUDA GRID CONFIGURATION                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Grid Size: 1024 x 1024 cells                               │
│  Block Size: 16 x 16 threads                                │
│  Grid Dimensions: (64, 64) blocks                           │
│                                                              │
│  ┌────┬────┬────┬────┬────┬────┐                            │
│  │B0,0│B1,0│B2,0│... │... │B63,0│  Each block = 16x16       │
│  ├────┼────┼────┼────┼────┼────┤  threads processing        │
│  │B0,1│B1,1│B2,1│... │... │B63,1│  16x16 cells              │
│  ├────┼────┼────┼────┼────┼────┤                            │
│  │... │... │... │... │... │... │  Total: 4096 blocks        │
│  ├────┼────┼────┼────┼────┼────┤  = 1,048,576 threads       │
│  │B0,63│B1,63│...│... │... │B63,63│                          │
│  └────┴────┴────┴────┴────┴────┘                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Comparison

### Key Differences

| Aspect | Python (NumPy) | CUDA |
|--------|----------------|------|
| **Parallelism** | SIMD via NumPy | Massive GPU parallelism |
| **Memory** | System RAM | GPU VRAM + Shared Memory |
| **Threads** | Single (vectorized) | Thousands concurrent |
| **Best For** | Small grids, prototyping | Large grids, production |

### Expected Speedup

| Grid Size | Python Time | CUDA Time | Speedup |
|-----------|-------------|-----------|---------|
| 32x32 | ~0.07 ms | ~0.01 ms | ~7x |
| 256x256 | ~0.34 ms | ~0.02 ms | ~17x |
| 1024x1024 | ~4.76 ms | ~0.1 ms | ~50x |
| 4096x4096 | ~80 ms | ~1 ms | ~80x |

---

## Toroidal Wrapping (Both Implementations)

```
┌─────────────────────────────────────────────────────────────┐
│                    TOROIDAL GRID TOPOLOGY                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  The grid wraps around like a donut (torus):                │
│                                                              │
│      ←───────────────────────────────────→                  │
│    ↑ ┌─────────────────────────────────────┐ ↑              │
│    │ │  (0,0)              ...       (W,0) │ │              │
│    │ │    ↖                             ↗  │ │              │
│    │ │      neighbors wrap to opposite     │ │              │
│    │ │      side of the grid               │ │              │
│    │ │    ↙                             ↘  │ │              │
│    │ │  (0,H)              ...       (W,H) │ │              │
│    ↓ └─────────────────────────────────────┘ ↓              │
│      ←───────────────────────────────────→                  │
│                                                              │
│  Formula: nx = (x + dx + width) % width                     │
│           ny = (y + dy + height) % height                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Using run.sh (Recommended)

```bash
# Show help
./run.sh help

# Run visual version (interactive)
./run.sh visual

# Run Python sequential version
./run.sh sequential 128 128 500

# Run CUDA parallel version
./run.sh cuda 1024 1024 1000

# Run all benchmarks
./run.sh benchmark

# Clean build artifacts
./run.sh clean
```

### Direct Execution

```bash
# Python Sequential
python src/python/game_of_life_sequential.py 256 256 500 0
python src/python/game_of_life_sequential.py --benchmark

# Visual
python src/visual/game_of_life_visual.py 200 150 5
```

---

## Visual Implementation (Pygame)

### File: `src/visual/game_of_life_visual.py`

### Features
- Real-time interactive visualization
- Draw/erase cells with mouse
- Adjustable simulation speed (1-60 fps)
- Zoom in/out and pan view
- Pre-built patterns (glider, glider gun)
- Status bar with generation count and live cells

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

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    VISUAL ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Events     │───→│  Game Logic  │───→│   Renderer   │   │
│  │  (Pygame)    │    │  (NumPy)     │    │  (Pygame)    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Mouse/Keys   │    │ Grid State   │    │ Screen       │   │
│  │ Input        │    │ (np.array)   │    │ Display      │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
│  Main Loop:                                                  │
│  1. Handle input events (mouse, keyboard)                   │
│  2. If running: compute next generation                     │
│  3. Render grid and UI                                      │
│  4. Control frame rate based on speed setting               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Show help
./run.sh help

# Run visual version (interactive)
./run.sh visual

# Run Python sequential version
./run.sh sequential 128 128 500

# Run CUDA parallel version
./run.sh cuda 1024 1024 1000

# Run all benchmarks
./run.sh benchmark
```

---

## Pattern Initializers

All implementations include:

1. **Random** - ~30% cells alive, uniformly distributed
2. **Glider** - Classic 5-cell spaceship pattern
3. **Gosper Glider Gun** - Produces new gliders periodically (requires 40x20 minimum)

```
Glider Pattern:          Gosper Glider Gun (partial):
    .#.                  ........................O...........
    ..#                  ......................O.O...........
    ###                  ............OO......OO............OO
                         ...........O...O....OO............OO
                         OO........O.....O...OO..............
```
