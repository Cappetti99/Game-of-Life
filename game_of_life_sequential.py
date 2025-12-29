"""
Conway's Game of Life - Sequential Python Implementation

Rules:
1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors lives on
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes alive (reproduction)

Usage: python game_of_life_sequential.py [width] [height] [generations] [visualize]
"""

import numpy as np
import time
import sys
import os


def count_neighbors(grid: np.ndarray, x: int, y: int, width: int, height: int) -> int:
    """
    Count live neighbors for a cell at position (x, y).
    Uses toroidal wrapping (edges connect to opposite sides).
    """
    count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue  # Skip the cell itself
            # Toroidal wrapping
            nx = (x + dx) % width
            ny = (y + dy) % height
            count += grid[ny, nx]
    return count


def game_of_life_step(current_grid: np.ndarray) -> np.ndarray:
    """
    Compute the next generation of the grid.
    Pure Python implementation (slow but clear).
    """
    height, width = current_grid.shape
    next_grid = np.zeros_like(current_grid)
    
    for y in range(height):
        for x in range(width):
            neighbors = count_neighbors(current_grid, x, y, width, height)
            current_state = current_grid[y, x]
            
            # Apply Game of Life rules
            if current_state == 1:
                # Live cell
                next_grid[y, x] = 1 if neighbors in (2, 3) else 0
            else:
                # Dead cell
                next_grid[y, x] = 1 if neighbors == 3 else 0
    
    return next_grid


def game_of_life_step_numpy(current_grid: np.ndarray) -> np.ndarray:
    """
    Compute the next generation using NumPy operations.
    Vectorized implementation (faster than pure Python).
    """
    # Count neighbors using convolution-like operation with roll
    neighbors = np.zeros_like(current_grid)
    
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            neighbors += np.roll(np.roll(current_grid, dy, axis=0), dx, axis=1)
    
    # Apply rules vectorized
    # Birth: dead cell with exactly 3 neighbors
    birth = (current_grid == 0) & (neighbors == 3)
    # Survival: live cell with 2 or 3 neighbors
    survive = (current_grid == 1) & ((neighbors == 2) | (neighbors == 3))
    
    return (birth | survive).astype(np.uint8)


def init_random(width: int, height: int, density: float = 0.3, seed: int | None = None) -> np.ndarray:
    """Initialize grid with random values."""
    if seed is not None:
        np.random.seed(seed)
    return (np.random.random((height, width)) < density).astype(np.uint8)


def init_glider(width: int, height: int, start_x: int = 1, start_y: int = 1) -> np.ndarray:
    """Initialize grid with a glider pattern."""
    grid = np.zeros((height, width), dtype=np.uint8)
    
    # Glider pattern
    #   #
    #     #
    # # # #
    if start_x + 2 < width and start_y + 2 < height:
        grid[start_y, start_x + 1] = 1
        grid[start_y + 1, start_x + 2] = 1
        grid[start_y + 2, start_x] = 1
        grid[start_y + 2, start_x + 1] = 1
        grid[start_y + 2, start_x + 2] = 1
    
    return grid


def init_glider_gun(width: int, height: int) -> np.ndarray:
    """Initialize grid with a Gosper Glider Gun."""
    grid = np.zeros((height, width), dtype=np.uint8)
    
    if width < 40 or height < 20:
        print("Grid too small for glider gun, need at least 40x20")
        return grid
    
    ox, oy = 1, 5  # Offset
    
    # Left square
    grid[oy + 0, ox + 0] = 1
    grid[oy + 0, ox + 1] = 1
    grid[oy + 1, ox + 0] = 1
    grid[oy + 1, ox + 1] = 1
    
    # Left part
    grid[oy + 0, ox + 10] = 1
    grid[oy + 1, ox + 10] = 1
    grid[oy + 2, ox + 10] = 1
    grid[oy - 1, ox + 11] = 1
    grid[oy + 3, ox + 11] = 1
    grid[oy - 2, ox + 12] = 1
    grid[oy + 4, ox + 12] = 1
    grid[oy - 2, ox + 13] = 1
    grid[oy + 4, ox + 13] = 1
    grid[oy + 1, ox + 14] = 1
    grid[oy - 1, ox + 15] = 1
    grid[oy + 3, ox + 15] = 1
    grid[oy + 0, ox + 16] = 1
    grid[oy + 1, ox + 16] = 1
    grid[oy + 2, ox + 16] = 1
    grid[oy + 1, ox + 17] = 1
    
    # Right part
    grid[oy - 2, ox + 20] = 1
    grid[oy - 1, ox + 20] = 1
    grid[oy + 0, ox + 20] = 1
    grid[oy - 2, ox + 21] = 1
    grid[oy - 1, ox + 21] = 1
    grid[oy + 0, ox + 21] = 1
    grid[oy - 3, ox + 22] = 1
    grid[oy + 1, ox + 22] = 1
    grid[oy - 4, ox + 24] = 1
    grid[oy - 3, ox + 24] = 1
    grid[oy + 1, ox + 24] = 1
    grid[oy + 2, ox + 24] = 1
    
    # Right square
    grid[oy - 2, ox + 34] = 1
    grid[oy - 2, ox + 35] = 1
    grid[oy - 1, ox + 34] = 1
    grid[oy - 1, ox + 35] = 1
    
    return grid


def print_grid(grid: np.ndarray) -> None:
    """Print the grid to console."""
    print("\033[H", end="")  # Move cursor to home position
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            print('#' if grid[y, x] else '.', end='')
        print()
    print()


def count_live_cells(grid: np.ndarray) -> int:
    """Count total live cells in the grid."""
    return np.sum(grid)


def run_simulation(width: int, height: int, generations: int, 
                   visualize: bool = False, use_numpy: bool = True,
                   seed: int | None = None) -> dict:
    """
    Run the Game of Life simulation.
    
    Args:
        width: Grid width
        height: Grid height
        generations: Number of generations to simulate
        visualize: Whether to print each generation
        use_numpy: Use vectorized NumPy (faster) or pure Python (slower)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with timing and statistics
    """
    # Initialize grid
    grid = init_random(width, height, density=0.3, seed=seed)
    
    print(f"Game of Life Sequential Python Implementation")
    print(f"Grid size: {width} x {height}")
    print(f"Generations: {generations}")
    print(f"Using {'NumPy vectorized' if use_numpy else 'pure Python'} implementation")
    print()
    
    initial_live = count_live_cells(grid)
    print(f"Initial live cells: {initial_live}")
    
    if visualize and width <= 80 and height <= 40:
        print("\033[2J", end="")  # Clear screen
        print_grid(grid)
    
    # Select step function
    step_func = game_of_life_step_numpy if use_numpy else game_of_life_step
    
    # Run simulation
    start_time = time.perf_counter()
    
    for gen in range(generations):
        grid = step_func(grid)
        
        if visualize and width <= 80 and height <= 40:
            print_grid(grid)
            print(f"Generation: {gen + 1}, Live cells: {count_live_cells(grid)}")
            time.sleep(0.1)
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    final_live = count_live_cells(grid)
    
    # Print results
    print(f"\nSimulation complete!")
    print(f"Final live cells: {final_live}")
    print(f"Total time: {elapsed_ms:.2f} ms")
    print(f"Time per generation: {elapsed_ms / generations:.4f} ms")
    print(f"Cells processed per second: {width * height * generations / elapsed_ms / 1000:.2f} million")
    
    return {
        "width": width,
        "height": height,
        "generations": generations,
        "initial_live_cells": initial_live,
        "final_live_cells": final_live,
        "total_time_ms": elapsed_ms,
        "time_per_generation_ms": elapsed_ms / generations,
        "cells_per_second_million": width * height * generations / elapsed_ms / 1000
    }


def benchmark(sizes: list[int] | None = None, generations: int = 100, seed: int = 42,
              verbose: bool = True) -> None:
    """
    Run benchmarks for different grid sizes.
    Useful for comparing with CUDA implementation.
    
    Args:
        sizes: List of grid sizes to test
        generations: Number of generations per test
        seed: Random seed for reproducibility
        verbose: If True, print detailed output for each size
    """
    if sizes is None:
        sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    print("=" * 60)
    print("BENCHMARK: Sequential Python (NumPy) Game of Life")
    print("=" * 60)
    print()
    
    results = []
    
    for size in sizes:
        if verbose:
            print(f"\n--- Grid size: {size}x{size} ---")
            result = run_simulation(size, size, generations, visualize=False, 
                                    use_numpy=True, seed=seed)
            print()
        else:
            # Quiet mode - just run and collect results
            grid = init_random(size, size, density=0.3, seed=seed)
            start_time = time.perf_counter()
            for _ in range(generations):
                grid = game_of_life_step_numpy(grid)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            result = {
                "width": size,
                "height": size,
                "generations": generations,
                "total_time_ms": elapsed_ms,
                "time_per_generation_ms": elapsed_ms / generations,
                "cells_per_second_million": size * size * generations / elapsed_ms / 1000
            }
            print(f"Size {size:>5}x{size:<5} done: {elapsed_ms:>10.2f} ms")
        
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
    
    # Save results to CSV for easy comparison
    csv_file = "benchmark_sequential.csv"
    with open(csv_file, "w") as f:
        f.write("size,generations,total_time_ms,time_per_generation_ms,cells_per_second_million\n")
        for r in results:
            f.write(f"{r['width']},{r['generations']},{r['total_time_ms']:.4f},"
                    f"{r['time_per_generation_ms']:.6f},{r['cells_per_second_million']:.4f}\n")
    print(f"\nResults saved to {csv_file}")


def main():
    # Check for benchmark mode first
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        verbose = True  # Set to False for quiet mode (only summary)
        benchmark(verbose=verbose)
        return
    
    # Default parameters
    width = 64
    height = 64
    generations = 100
    visualize = False
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        width = int(sys.argv[1])
    if len(sys.argv) > 2:
        height = int(sys.argv[2])
    if len(sys.argv) > 3:
        generations = int(sys.argv[3])
    if len(sys.argv) > 4:
        visualize = bool(int(sys.argv[4]))
    
    # Run simulation
    run_simulation(width, height, generations, visualize=visualize, use_numpy=True, seed=42)


if __name__ == "__main__":
    main()
