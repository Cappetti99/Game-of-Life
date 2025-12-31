"""
Conway's Game of Life - Sequential Python Implementation

Usage: python game_of_life_sequential.py [width] [height] [generations] [visualize]
       python game_of_life_sequential.py --benchmark
"""

import numpy as np
import time
import sys


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


def run_simulation(width: int, height: int, generations: int, seed: int = 42) -> dict:
    """Run the Game of Life simulation."""
    grid = init_random(width, height, density=0.3, seed=seed)
    
    print(f"Game of Life Sequential Python Implementation")
    print(f"Grid size: {width} x {height}")
    print(f"Generations: {generations}")
    print()
    print(f"Initial live cells: {np.sum(grid)}")
    
    start_time = time.perf_counter()
    for _ in range(generations):
        grid = game_of_life_step(grid)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    print(f"Final live cells: {np.sum(grid)}")
    print(f"\nSimulation complete!")
    print(f"Total time: {elapsed_ms:.2f} ms")
    print(f"Time per generation: {elapsed_ms / generations:.4f} ms")
    print(f"Cells processed per second: {width * height * generations / elapsed_ms / 1000:.2f} million")
    
    return {
        "width": width,
        "height": height,
        "generations": generations,
        "total_time_ms": elapsed_ms,
        "time_per_generation_ms": elapsed_ms / generations,
        "cells_per_second_million": width * height * generations / elapsed_ms / 1000
    }


def benchmark(generations: int = 100, seed: int = 42) -> None:
    """Run benchmarks for different grid sizes."""
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    print("=" * 60)
    print("BENCHMARK: Sequential Python (NumPy) Game of Life")
    print("=" * 60)
    print()
    
    results = []
    
    for size in sizes:
        print(f"\n--- Grid size: {size}x{size} ---")
        result = run_simulation(size, size, generations, seed=seed)
        print()
        results.append(result)
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Size':>10} | {'Total (ms)':>12} | {'Per Gen (ms)':>14} | {'M cells/s':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['width']:>10} | {r['total_time_ms']:>12.2f} | "
              f"{r['time_per_generation_ms']:>14.4f} | {r['cells_per_second_million']:>12.2f}")
    print("=" * 60)
    
    # Save results to CSV
    with open("benchmark_sequential.csv", "w") as f:
        f.write("size,generations,total_time_ms,time_per_generation_ms,cells_per_second_million\n")
        for r in results:
            f.write(f"{r['width']},{r['generations']},{r['total_time_ms']:.4f},"
                    f"{r['time_per_generation_ms']:.6f},{r['cells_per_second_million']:.4f}\n")
    print(f"\nResults saved to benchmark_sequential.csv")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark()
        return
    
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    run_simulation(width, height, generations)


if __name__ == "__main__":
    main()
