#!/usr/bin/env python3
"""
Speedup Evaluation Script
Evaluates the speedup of CUDA implementation with different thread configurations
compared to the sequential implementation.
"""

import subprocess
import os
import sys
import csv
import time
from pathlib import Path

GRID_SIZES = [128, 256, 512, 1024, 2048]
BLOCK_SIZES = [4, 8, 16, 32]  # Different thread configurations
GENERATIONS = 100
SEED = 42

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
CUDA_SRC = PROJECT_ROOT / "src/cuda/game_of_life.cu"


def run_sequential(size, generations):
    """Run sequential implementation and return execution time in ms."""
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "game_of_life_sequential.py"),
             str(size), str(size), str(generations)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output for timing (handles both single run and averaged output)
        for line in result.stdout.split('\n'):
            # Check for mean time first (from averaged runs)
            if 'Mean time:' in line:
                # Extract "Mean time: 71.08 ± 0.42 ms"
                parts = line.split(':')[1].strip().split()
                time_ms = float(parts[0])
                return time_ms
            # Fallback to single run format
            elif 'Total time:' in line:
                time_ms = float(line.split(':')[1].strip().split()[0])
                return time_ms
        
        print(f"Warning: Could not parse sequential timing for size {size}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running sequential: {e}")
        return None


def compile_cuda(block_size):
    """Compile CUDA code with specific block size."""
    BUILD_DIR.mkdir(exist_ok=True)
    cuda_bin = BUILD_DIR / f"game_of_life_bs{block_size}"
    
    print(f"  Compiling CUDA with BLOCK_SIZE={block_size}...")
    
    compile_cmd = [
        "nvcc",
        "-o", str(cuda_bin),
        "-DBLOCK_SIZE=" + str(block_size),
        str(CUDA_SRC)
    ]
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        return cuda_bin
    except subprocess.CalledProcessError as e:
        print(f"Error compiling CUDA: {e}")
        print(f"stderr: {e.stderr}")
        return None


def run_cuda(cuda_bin, size, generations):
    """Run CUDA implementation and return execution time in ms."""
    try:
        result = subprocess.run(
            [str(cuda_bin), str(size), str(size), str(generations), "0", str(SEED)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output for timing (handles both single run and averaged output)
        for line in result.stdout.split('\n'):
            # Check for mean time first (from averaged runs)
            if 'Mean time:' in line:
                # Extract "Mean time: 2.84 ± 0.12 ms"
                parts = line.split(':')[1].strip().split()
                time_ms = float(parts[0])
                return time_ms
            # Fallback to single run format
            elif 'Total time:' in line:
                time_ms = float(line.split(':')[1].strip().split()[0])
                return time_ms
        
        print(f"Warning: Could not parse CUDA timing")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running CUDA: {e}")
        return None


def main():
    print("=" * 80)
    print("SPEEDUP EVALUATION: CUDA vs Sequential")
    print("Varying thread count (block size) in CUDA implementation")
    print("=" * 80)
    print()
    
    if not subprocess.run(["which", "nvcc"], capture_output=True).returncode == 0:
        print("Error: nvcc not found. CUDA toolkit required.")
        sys.exit(1)
    
    BENCHMARK_DIR.mkdir(exist_ok=True)
    
    results = []
    
    for size in GRID_SIZES:
        print(f"\n{'=' * 80}")
        print(f"Grid Size: {size}x{size}, Generations: {GENERATIONS}")
        print(f"{'=' * 80}")
        
        print(f"Running sequential version...")
        seq_time = run_sequential(size, GENERATIONS)
        
        if seq_time is None:
            print(f"Skipping size {size} due to sequential error")
            continue
        
        print(f"  Sequential time: {seq_time:.2f} ms")
        
        size_results = {
            'size': size,
            'sequential_ms': seq_time,
            'cuda_times': {},
            'speedups': {}
        }
        
        for block_size in BLOCK_SIZES:
            print(f"\nBlock size: {block_size}x{block_size} (threads per block: {block_size**2})")
            
            cuda_bin = compile_cuda(block_size)
            if cuda_bin is None:
                continue
            
            print(f"  Running CUDA version...")
            cuda_time = run_cuda(cuda_bin, size, GENERATIONS)
            
            if cuda_time is None:
                continue
            
            speedup = seq_time / cuda_time
            
            print(f"  CUDA time: {cuda_time:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            size_results['cuda_times'][block_size] = cuda_time
            size_results['speedups'][block_size] = speedup
        
        results.append(size_results)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    
    # Header
    header = f"{'Grid Size':<12} | {'Sequential':<12} |"
    for bs in BLOCK_SIZES:
        header += f" BS={bs:<3} Time | BS={bs:<3} Speedup |"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for r in results:
        row = f"{r['size']:>10}  | {r['sequential_ms']:>10.2f} ms |"
        for bs in BLOCK_SIZES:
            if bs in r['cuda_times']:
                row += f" {r['cuda_times'][bs]:>9.2f} ms | {r['speedups'][bs]:>11.2f}x |"
            else:
                row += f" {'N/A':>9} | {'N/A':>11} |"
        print(row)
    
    print("=" * 80)
    
    # Save detailed CSV
    csv_path = BENCHMARK_DIR / "speedup_evaluation.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header_row = ['grid_size', 'sequential_time_ms']
        for bs in BLOCK_SIZES:
            header_row.extend([f'cuda_bs{bs}_time_ms', f'speedup_bs{bs}'])
        writer.writerow(header_row)
        
        for r in results:
            row = [r['size'], f"{r['sequential_ms']:.4f}"]
            for bs in BLOCK_SIZES:
                if bs in r['cuda_times']:
                    row.extend([f"{r['cuda_times'][bs]:.4f}", f"{r['speedups'][bs]:.4f}"])
                else:
                    row.extend(['', ''])
            writer.writerow(row)
    
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Save summary table
    summary_path = BENCHMARK_DIR / "speedup_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SPEEDUP EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generations per run: {GENERATIONS}\n")
        f.write(f"Block sizes tested: {BLOCK_SIZES}\n")
        f.write(f"Grid sizes tested: {GRID_SIZES}\n\n")
        
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in results:
            row = f"{r['size']:>10}  | {r['sequential_ms']:>10.2f} ms |"
            for bs in BLOCK_SIZES:
                if bs in r['cuda_times']:
                    row += f" {r['cuda_times'][bs]:>9.2f} ms | {r['speedups'][bs]:>11.2f}x |"
                else:
                    row += f" {'N/A':>9} | {'N/A':>11} |"
            f.write(row + "\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary table saved to: {summary_path}")
    
    # Find best configuration for each size
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION FOR EACH GRID SIZE")
    print("=" * 80)
    for r in results:
        if r['speedups']:
            best_bs = max(r['speedups'].items(), key=lambda x: x[1])
            print(f"Grid {r['size']:>4}x{r['size']:<4}: Block Size {best_bs[0]:>2}x{best_bs[0]:<2} "
                  f"(Speedup: {best_bs[1]:.2f}x, Time: {r['cuda_times'][best_bs[0]]:.2f} ms)")
    print("=" * 80)


if __name__ == "__main__":
    main()
