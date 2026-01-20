"""
Conway's Game of Life - Sequential Python Implementation

Sequential (CPU) implementation of Conway's Game of Life.
Used as a baseline for performance comparisons with parallel versions.

"""

import numpy as np
import time
import sys
import gc

# Optional SciPy import (used only if available)
try:
    from scipy import stats  # unused; kept for completeness
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Fast neighbor count via 2D convolution (if available)
try:
    from scipy.ndimage import convolve
    HAS_SCIPY_NDIMAGE = True
except ImportError:
    HAS_SCIPY_NDIMAGE = False


def game_of_life_step(grid: np.ndarray) -> np.ndarray:
    """Compute one Game of Life step (0=dead, 1=alive)."""

    # Use convolution for neighbor counting when SciPy is present
    if HAS_SCIPY_NDIMAGE:
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)  # 3x3 stencil (no center)
        neighbors = convolve(grid, kernel, mode='wrap')  # toroidal boundaries
    else:
        neighbors = np.zeros_like(grid)  # fallback accumulator
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # skip the center cell
                neighbors += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)  # shift + sum

    birth = (grid == 0) & (neighbors == 3)  # B3 rule
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))  # S23 rule

    return (birth | survive).astype(np.uint8)  # next generation as uint8


def init_random(width: int, height: int,
                density: float = 0.3,
                seed: int = 42) -> np.ndarray:
    """Create a random initial grid with given live-cell density."""
    np.random.seed(seed)  # deterministic runs
    return (np.random.random((height, width)) < density).astype(np.uint8)  # Bernoulli field


def run_simulation(width: int,
                   height: int,
                   generations: int,
                   seed: int = 42,
                   warmup_runs: int = 0,
                   measure_runs: int = 1) -> dict:
    """Run the simulation and collect timing statistics."""

    grid = init_random(width, height, density=0.3, seed=seed)  # initial state

    print("Game of Life (Sequential)")
    print(f"Grid: {width}x{height} | Generations: {generations}")
    print("Backend: scipy.ndimage.convolve" if HAS_SCIPY_NDIMAGE else "Backend: numpy.roll (fallback)")

    # Warm-up runs (reduce cache/frequency effects)
    if warmup_runs > 0:
        for _ in range(warmup_runs):
            temp_grid = grid.copy()  # keep original grid intact
            for _ in range(generations):
                temp_grid = game_of_life_step(temp_grid)  # advance state

    gc.collect()  # clean before timing
    gc.disable()  # reduce noise during timing

    timings_ms = []  # per-run elapsed times

    for _ in range(measure_runs):
        temp_grid = grid.copy()  # reset to same initial state
        start_time = time.perf_counter()  # high-resolution timer

        for _ in range(generations):
            temp_grid = game_of_life_step(temp_grid)  # main loop

        elapsed_ms = (time.perf_counter() - start_time) * 1000  # seconds -> ms
        timings_ms.append(elapsed_ms)  # store run time

    gc.enable()  # restore GC

    timings = np.array(timings_ms)  # vectorize stats
    mean_time = np.mean(timings)  # average runtime (ms)
    std_time = np.std(timings, ddof=1) if measure_runs > 1 else 0.0  # sample std

    throughput = width * height * generations / timings / 1000  # MCells/s (cells/ms -> MCells/s)
    mean_throughput = np.mean(throughput)  # average throughput
    std_throughput = np.std(throughput, ddof=1) if measure_runs > 1 else 0.0  # sample std

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
        "cv_percent": (std_time / mean_time * 100) if mean_time > 0 else 0.0  # coefficient of variation
    }


def benchmark(generations: int = 100,
              seed: int = 42,
              warmup_runs: int = 2,
              measure_runs: int = 5) -> None:
    """Benchmark multiple grid sizes and export CSV."""

    sizes = [32, 64, 128, 256, 512, 1024, 2048]  # tested sizes

    print("\nStarting Sequential Benchmark")
    print("=" * 60)

    results = []  # accumulate dict results
    for size in sizes:
        result = run_simulation(size, size, generations,
                                seed=seed,
                                warmup_runs=warmup_runs,
                                measure_runs=measure_runs)
        results.append(result)  # store result row

    filename = "benchmark_sequential.csv"  # output path
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
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark()  # sweep sizes + CSV
        return

    width = int(sys.argv[1]) if len(sys.argv) > 1 else 64  # CLI or default
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 64  # CLI or default
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 100  # CLI or default

    run_simulation(width, height, generations,
                   warmup_runs=0,
                   measure_runs=1)  # single-run quick test


if __name__ == "__main__":
    main()  # entry point