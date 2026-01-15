"""
Conway's Game of Life - Sequential Python Implementation

Usage: python game_of_life_sequential.py [width] [height] [generations] [visualize]
       python game_of_life_sequential.py --benchmark
"""

import numpy as np
import time
import sys

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def game_of_life_step(grid: np.ndarray) -> np.ndarray:
    """Compute the next generation using NumPy vectorized operations."""
    neighbors = np.zeros_like(grid)
    
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            neighbors += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
    
    birth = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    
    return (birth | survive).astype(np.uint8)


def init_random(width: int, height: int, density: float = 0.3, seed: int = 42) -> np.ndarray:
    """Initialize grid with random values."""
    np.random.seed(seed)
    return (np.random.random((height, width)) < density).astype(np.uint8)


def run_simulation(width: int, height: int, generations: int, seed: int = 42, warmup_runs: int = 0, measure_runs: int = 1) -> dict:
    """Run the Game of Life simulation with optional multiple runs for averaging."""
    grid = init_random(width, height, density=0.3, seed=seed)
    
    print(f"Game of Life Sequential Python Implementation")
    print(f"Grid size: {width} x {height}")
    print(f"Generations: {generations}")
    if warmup_runs > 0 or measure_runs > 1:
        print(f"Warmup runs: {warmup_runs}, Measurement runs: {measure_runs}")
    print()
    print(f"Initial live cells: {np.sum(grid)}")
    
    # Warmup runs (not measured)
    for i in range(warmup_runs):
        temp_grid = grid.copy()
        for _ in range(generations):
            temp_grid = game_of_life_step(temp_grid)
        if warmup_runs > 1:
            print(f"  Warmup {i+1}/{warmup_runs} complete", end='\r')
    if warmup_runs > 0:
        print(f"  Warmup complete ({warmup_runs} runs)   ")
    
    # Measurement runs
    timings_ms = []
    for run in range(measure_runs):
        temp_grid = grid.copy()
        start_time = time.perf_counter()
        for _ in range(generations):
            temp_grid = game_of_life_step(temp_grid)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timings_ms.append(elapsed_ms)
        if measure_runs > 1:
            print(f"  Run {run+1}/{measure_runs}: {elapsed_ms:.2f} ms", end='\r')
        grid = temp_grid  # Keep final state
    
    if measure_runs > 1:
        print()  # New line after progress
    
    # Calculate statistics
    timings_array = np.array(timings_ms)
    mean_time = np.mean(timings_array)
    std_time = np.std(timings_array, ddof=1) if measure_runs > 1 else 0.0
    median_time = np.median(timings_array)
    min_time = np.min(timings_array)
    max_time = np.max(timings_array)
    
    # Calculate throughput statistics
    throughput_array = width * height * generations / timings_array / 1000
    mean_throughput = np.mean(throughput_array)
    std_throughput = np.std(throughput_array, ddof=1) if measure_runs > 1 else 0.0
    
    # Calculate 95% confidence interval
    ci_range = 0.0
    if measure_runs > 1 and HAS_SCIPY:
        from scipy import stats as scipy_stats
        ci_95 = scipy_stats.t.interval(0.95, measure_runs - 1, 
                                       loc=mean_time, 
                                       scale=scipy_stats.sem(timings_array))
        ci_range = ci_95[1] - mean_time
    
    print(f"Final live cells: {np.sum(grid)}")
    print(f"\nSimulation complete!")
    if measure_runs > 1:
        print(f"Mean time: {mean_time:.2f} ± {std_time:.2f} ms (n={measure_runs})")
        print(f"Time range: [{min_time:.2f}, {max_time:.2f}] ms")
        print(f"Time per generation: {mean_time / generations:.4f} ms")
        print(f"Cells processed per second: {mean_throughput:.2f} ± {std_throughput:.2f} million")
    else:
        print(f"Total time: {mean_time:.2f} ms")
        print(f"Time per generation: {mean_time / generations:.4f} ms")
        print(f"Cells processed per second: {mean_throughput:.2f} million")
    
    return {
        "width": width,
        "height": height,
        "generations": generations,
        "runs": measure_runs,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "median_time_ms": median_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "ci_95_range_ms": ci_range,
        "mean_throughput_mcells_s": mean_throughput,
        "std_throughput_mcells_s": std_throughput,
        "cv_percent": (std_time / mean_time * 100) if mean_time > 0 else 0.0
    }


def benchmark(generations: int = 100, seed: int = 42, warmup_runs: int = 2, measure_runs: int = 10) -> None:
    """Run benchmarks for different grid sizes with multiple runs for statistical averaging."""
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    print("=" * 80)
    print("BENCHMARK: Sequential Python (NumPy) Game of Life")
    print(f"Configuration: {warmup_runs} warmup runs + {measure_runs} measurement runs per size")
    print("=" * 80)
    print()
    
    results = []
    
    for size in sizes:
        print(f"\n{'='*80}")
        print(f"Grid size: {size}x{size}")
        print(f"{'='*80}")
        result = run_simulation(size, size, generations, seed=seed, 
                               warmup_runs=warmup_runs, measure_runs=measure_runs)
        print()
        results.append(result)
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY (Mean ± Std Dev)")
    print("=" * 80)
    print(f"{'Size':>10} | {'Mean (ms)':>15} | {'Per Gen (ms)':>14} | {'M cells/s':>15} | {'CV%':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['width']:>10} | {r['mean_time_ms']:>10.2f}±{r['std_time_ms']:>4.2f} | "
              f"{r['mean_time_ms']/r['generations']:>14.4f} | "
              f"{r['mean_throughput_mcells_s']:>10.2f}±{r['std_throughput_mcells_s']:>4.2f} | "
              f"{r['cv_percent']:>6.2f}")
    print("=" * 80)
    
    # Save results to CSV with statistical measures
    with open("benchmark_sequential.csv", "w") as f:
        f.write("size,generations,runs,mean_time_ms,std_time_ms,median_time_ms,min_time_ms,max_time_ms,"
                "ci_95_range_ms,mean_throughput_mcells_s,std_throughput_mcells_s,cv_percent\n")
        for r in results:
            f.write(f"{r['width']},{r['generations']},{r['runs']},"
                    f"{r['mean_time_ms']:.4f},{r['std_time_ms']:.4f},{r['median_time_ms']:.4f},"
                    f"{r['min_time_ms']:.4f},{r['max_time_ms']:.4f},{r['ci_95_range_ms']:.4f},"
                    f"{r['mean_throughput_mcells_s']:.4f},{r['std_throughput_mcells_s']:.4f},"
                    f"{r['cv_percent']:.4f}\n")
    print(f"\nResults saved to benchmark_sequential.csv")
    print(f"Note: CV% = Coefficient of Variation (lower is better, <5% is excellent)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark()
        return
    
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Single run mode (no averaging) for regular usage
    run_simulation(width, height, generations, warmup_runs=0, measure_runs=1)


if __name__ == "__main__":
    main()
