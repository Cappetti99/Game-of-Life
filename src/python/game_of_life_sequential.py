"""
Conway's Game of Life - Sequential Python Implementation

Sequential (CPU) implementation of Conway's Game of Life.
Used as a baseline for performance comparisons with parallel versions.

Main ideas:
- Use NumPy vectorization instead of Python loops over cells.
- Count neighbors using scipy.ndimage.convolve when available.
- Include a simple but solid benchmarking setup.
"""

import numpy as np
import time
import sys
import gc

# Try to import SciPy.
# If available, we can use optimized C implementations.
try:
    from scipy import stats  # not used directly, kept for completeness
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# This is the important one: optimized 2D convolution
try:
    from scipy.ndimage import convolve
    HAS_SCIPY_NDIMAGE = True
except ImportError:
    HAS_SCIPY_NDIMAGE = False


def game_of_life_step(grid: np.ndarray) -> np.ndarray:
    """
    Compute one step of the Game of Life.
    grid contains 0 (dead) or 1 (alive).
    """

    # --- Neighbor counting ---
    # If SciPy is available, use a 3x3 convolution kernel.
    # mode='wrap' implements toroidal boundary conditions.
    if HAS_SCIPY_NDIMAGE:
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        neighbors = convolve(grid, kernel, mode='wrap')
    else:
        # Fallback: sum 8 shifted versions of the grid.
        neighbors = np.zeros_like(grid)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                neighbors += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)

    # --- Apply Game of Life rules (B3/S23) ---
    # Birth: dead cell with exactly 3 neighbors
    birth = (grid == 0) & (neighbors == 3)

    # Survival: live cell with 2 or 3 neighbors
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))

    # Combine conditions and return next state
    return (birth | survive).astype(np.uint8)


def init_random(width: int, height: int,
                density: float = 0.3,
                seed: int = 42) -> np.ndarray:
    """
    Initialize the grid with random live cells.
    """
    np.random.seed(seed)
    return (np.random.random((height, width)) < density).astype(np.uint8)


def run_simulation(width: int,
                   height: int,
                   generations: int,
                   seed: int = 42,
                   warmup_runs: int = 0,
                   measure_runs: int = 1) -> dict:
    """
    Run the simulation and measure performance.
    """

    # Initial grid
    grid = init_random(width, height, density=0.3, seed=seed)

    print("Game of Life (Sequential)")
    print(f"Grid: {width}x{height} | Generations: {generations}")
    if HAS_SCIPY_NDIMAGE:
        print("Backend: scipy.ndimage.convolve")
    else:
        print("Backend: numpy.roll (fallback)")

    # --- Warmup phase ---
    # Helps stabilize cache and CPU frequency before timing.
    if warmup_runs > 0:
        for _ in range(warmup_runs):
            temp_grid = grid.copy()
            for _ in range(generations):
                temp_grid = game_of_life_step(temp_grid)

    # --- Measurement phase ---
    # Disable GC to reduce timing noise.
    gc.collect()
    gc.disable()

    timings_ms = []

    for _ in range(measure_runs):
        temp_grid = grid.copy()
        start_time = time.perf_counter()

        for _ in range(generations):
            temp_grid = game_of_life_step(temp_grid)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timings_ms.append(elapsed_ms)

    gc.enable()

    # --- Statistics ---
    timings = np.array(timings_ms)
    mean_time = np.mean(timings)
    std_time = np.std(timings, ddof=1) if measure_runs > 1 else 0.0

    # Throughput: million cell updates per second
    throughput = width * height * generations / timings / 1000
    mean_throughput = np.mean(throughput)
    std_throughput = np.std(throughput, ddof=1) if measure_runs > 1 else 0.0

    print(f"Results ({measure_runs} runs):")
    if measure_runs > 1:
        print(f"  Mean time:  {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"  Throughput:{mean_throughput:.2f} ± {std_throughput:.2f} MCells/s")
    else:
        print(f"  Time:      {mean_time:.2f} ms")
        print(f"  Throughput:{mean_throughput:.2f} MCells/s")
    print("-" * 40)

    return {
        "width": width,
        "height": height,
        "generations": generations,
        "runs": measure_runs,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "mean_throughput_mcells_s": mean_throughput,
        "std_throughput_mcells_s": std_throughput,
        "cv_percent": (std_time / mean_time * 100) if mean_time > 0 else 0.0
    }


def benchmark(generations: int = 100,
              seed: int = 42,
              warmup_runs: int = 2,
              measure_runs: int = 5) -> None:
    """
    Run benchmark on different grid sizes.
    """

    sizes = [32, 64, 128, 256, 512, 1024, 2048]

    print("\nStarting Sequential Benchmark")
    print("=" * 60)

    results = []
    for size in sizes:
        result = run_simulation(size, size, generations,
                                seed=seed,
                                warmup_runs=warmup_runs,
                                measure_runs=measure_runs)
        results.append(result)

    # Save results for external plotting
    filename = "benchmark_sequential.csv"
    with open(filename, "w") as f:
        f.write(
            "size,generations,runs,mean_time_ms,std_time_ms,"
            "mean_throughput_mcells_s,std_throughput_mcells_s,cv_percent\n"
        )
        for r in results:
            f.write(
                f"{r['width']},{r['generations']},{r['runs']},"
                f"{r['mean_time_ms']:.4f},{r['std_time_ms']:.4f},"
                f"{r['mean_throughput_mcells_s']:.4f},{r['std_throughput_mcells_s']:.4f},"
                f"{r['cv_percent']:.4f}\n"
            )

    print(f"\nBenchmark complete. Results saved to {filename}")


def main():
    # Benchmark mode
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark()
        return

    # Default parameters
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    run_simulation(width, height, generations,
                   warmup_runs=0,
                   measure_runs=1)


if __name__ == "__main__":
    main()
