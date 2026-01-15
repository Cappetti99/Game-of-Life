#!/usr/bin/env python3
"""
Performance Analysis & Visualization Suite - Academic Style
Publication-quality plots for Game of Life performance analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import numpy as np
import sys
from pathlib import Path

# Academic/Publication style settings
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
})

# Professional color palette (colorblind-friendly)
COLORS = {
    'cuda': '#0173B2',      # Blue
    'python': '#DE8F05',    # Orange
    'optimal': '#029E73',   # Green
    'baseline': '#CC78BC',  # Purple
    'gray': '#949494',      # Gray
}

def load_data():
    """Load all benchmark data with support for both old and new CSV formats"""
    data = {}
    
    # Try multiple paths for flexibility
    paths = {
        'sequential': ['benchmarks/benchmark_sequential.csv', 
                      '../../benchmarks/benchmark_sequential.csv',
                      '../benchmarks/benchmark_sequential.csv'],
        'cuda': ['benchmarks/benchmark_cuda.csv',
                '../../benchmarks/benchmark_cuda.csv',
                '../benchmarks/benchmark_cuda.csv'],
        'block_sizes': ['benchmarks/block_size_comparison.csv',
                       '../../benchmarks/block_size_comparison.csv',
                       '../benchmarks/block_size_comparison.csv']
    }
    
    for key, path_list in paths.items():
        for path in path_list:
            if Path(path).exists():
                try:
                    df = pd.read_csv(path)
                    
                    # Handle new format with statistics (mean_time_ms, std_time_ms, etc.)
                    # or old format (total_time_ms, time_ms)
                    
                    # Normalize column names for sequential data
                    if key == 'sequential':
                        if 'size' in df.columns:
                            df = df.rename(columns={'size': 'grid_size'})
                        
                        # New format with statistics
                        if 'mean_time_ms' in df.columns:
                            df = df.rename(columns={
                                'mean_time_ms': 'time_ms',
                                'mean_throughput_mcells_s': 'throughput_mcells_s'
                            })
                            # Calculate time_per_gen_ms if generations column exists
                            if 'generations' in df.columns and 'time_per_gen_ms' not in df.columns:
                                df['time_per_gen_ms'] = df['time_ms'] / df['generations']
                            # Keep std columns for error bars
                        # Old format
                        elif 'total_time_ms' in df.columns:
                            df = df.rename(columns={
                                'total_time_ms': 'time_ms',
                                'time_per_generation_ms': 'time_per_gen_ms',
                                'cells_per_second_million': 'throughput_mcells_s'
                            })
                    
                    # Normalize column names for cuda data
                    if key == 'cuda':
                        if 'size' in df.columns:
                            df = df.rename(columns={'size': 'grid_size'})
                        
                        # New format with statistics
                        if 'mean_time_ms' in df.columns:
                            df = df.rename(columns={
                                'mean_time_ms': 'time_ms',
                                'mean_throughput_mcells_s': 'throughput_mcells_s'
                            })
                            # Calculate time_per_gen_ms if generations column exists
                            if 'generations' in df.columns and 'time_per_gen_ms' not in df.columns:
                                df['time_per_gen_ms'] = df['time_ms'] / df['generations']
                        # Old format
                        elif 'total_time_ms' in df.columns:
                            df = df.rename(columns={
                                'total_time_ms': 'time_ms',
                                'time_per_generation_ms': 'time_per_gen_ms',
                                'cells_per_second_million': 'throughput_mcells_s'
                            })
                    
                    # Block sizes data
                    if key == 'block_sizes':
                        # New format with statistics
                        if 'mean_time_ms' in df.columns:
                            df = df.rename(columns={
                                'mean_time_ms': 'time_ms',
                                'mean_throughput_mcells_s': 'throughput_mcells_s'
                            })
                        # Old format - no renaming needed, already has time_ms and throughput_mcells_s
                    
                    data[key] = df
                    print(f"Loaded {key}: {len(df)} records from {path}")
                    break
                except Exception as e:
                    print(f"Warning: Error loading {path}: {e}")
        
        if key not in data:
            print(f"Warning: {key} data not found")
    
    return data

def calculate_metrics(df_seq, df_cuda, df_block):
    """Calculate all performance metrics"""
    metrics = {}
    
    # CUDA vs Python speedup analysis
    if df_seq is not None and df_cuda is not None:
        common_sizes = sorted(set(df_seq['grid_size'].unique()) & set(df_cuda['grid_size'].unique()))
        
        speedups = []
        efficiencies = []
        
        for size in common_sizes:
            seq_time = df_seq[df_seq['grid_size'] == size]['time_per_gen_ms'].values[0]
            cuda_time = df_cuda[df_cuda['grid_size'] == size]['time_per_gen_ms'].values[0]
            
            speedup = seq_time / cuda_time
            speedups.append(speedup)
            
            # Theoretical efficiency (assuming GPU has ~1000 cores)
            theoretical_speedup = 1000
            efficiency = (speedup / theoretical_speedup) * 100
            efficiencies.append(efficiency)
        
        metrics['common_sizes'] = common_sizes
        metrics['speedups'] = speedups
        metrics['efficiencies'] = efficiencies
        
        # Find break-even point (where CUDA beats Python)
        breakeven_idx = next((i for i, s in enumerate(speedups) if s > 1), None)
        metrics['breakeven_size'] = common_sizes[breakeven_idx] if breakeven_idx is not None else None
    
    # Strong scaling analysis (fixed grid, varying block sizes)
    if df_block is not None:
        # Pick largest grid size for strong scaling
        grid_sizes = sorted(df_block['grid_size'].unique())
        largest_grid = grid_sizes[-1]
        
        strong_data = df_block[df_block['grid_size'] == largest_grid].copy()
        strong_data = strong_data.sort_values('block_size')
        
        # Calculate speedup relative to block_size=1
        baseline_time = strong_data[strong_data['block_size'] == 1]['time_ms'].values[0]
        strong_data['speedup'] = baseline_time / strong_data['time_ms']
        strong_data['ideal_speedup'] = strong_data['block_size'] ** 2  # Ideal linear scaling
        strong_data['efficiency'] = (strong_data['speedup'] / strong_data['ideal_speedup']) * 100
        
        metrics['strong_scaling'] = strong_data
    
    # Weak scaling analysis (proportional increase in grid and threads)
    if df_block is not None:
        block_sizes = sorted(df_block['block_size'].unique())
        weak_scaling = []
        
        for bs in block_sizes:
            # Get performance for each grid size at this block size
            data = df_block[df_block['block_size'] == bs].sort_values('grid_size')
            if len(data) > 0:
                # Calculate time per cell
                data = data.copy()
                data['time_per_cell'] = data['time_ms'] / (data['grid_size'] ** 2 * data['generations'])
                weak_scaling.append({
                    'block_size': bs,
                    'data': data
                })
        
        metrics['weak_scaling'] = weak_scaling
    
    return metrics

def create_academic_plots(df_seq, df_cuda, df_block, metrics, output_dir):
    """Create individual publication-quality academic plots"""
    saved_files = []
    
    # Plot 1: Speedup vs Grid Size (Log-Log)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    if 'common_sizes' in metrics and 'speedups' in metrics:
        sizes = metrics['common_sizes']
        speedups = metrics['speedups']
        
        ax1.plot(sizes, speedups, 'o-', color=COLORS['cuda'], 
                linewidth=2, markersize=7, label='Measured Speedup')
        
        # Ideal linear scaling reference
        ideal = np.array(sizes) / sizes[0]
        ax1.plot(sizes, ideal, '--', color=COLORS['gray'], 
                linewidth=1.5, alpha=0.6, label='Linear Scaling')
        
        ax1.set_xlabel('Grid Size (N×N)')
        ax1.set_ylabel('Speedup (CUDA vs Python)')
        ax1.set_title('Performance Speedup', fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.legend(frameon=True, fancybox=False, edgecolor='black')
        ax1.grid(True, which='both', alpha=0.3)
        
        fig1.tight_layout()
        output_path = output_dir / '1_speedup_analysis.png'
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        plt.close(fig1)
    
    # Plot 2: Throughput Comparison
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    if df_seq is not None and df_cuda is not None:
        sizes = sorted(set(df_seq['grid_size'].unique()) & set(df_cuda['grid_size'].unique()))
        
        python_tp = [df_seq[df_seq['grid_size'] == s]['throughput_mcells_s'].values[0] for s in sizes]
        cuda_tp = [df_cuda[df_cuda['grid_size'] == s]['throughput_mcells_s'].values[0] for s in sizes]
        
        ax2.plot(sizes, python_tp, 's-', color=COLORS['python'], 
                linewidth=2, markersize=6, label='Python (NumPy)')
        ax2.plot(sizes, cuda_tp, 'o-', color=COLORS['cuda'], 
                linewidth=2, markersize=6, label='CUDA (GPU)')
        
        ax2.set_xlabel('Grid Size (N×N)')
        ax2.set_ylabel('Throughput (M cells/s)')
        ax2.set_title('Absolute Throughput', fontweight='bold')
        ax2.set_yscale('log')
        ax2.set_xscale('log', base=2)
        ax2.legend(frameon=True, fancybox=False, edgecolor='black')
        ax2.grid(True, which='both', alpha=0.3)
        
        fig2.tight_layout()
        output_path = output_dir / '2_throughput_comparison.png'
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        plt.close(fig2)
    
    # Plot 3: Block Size Optimization
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    if df_block is not None:
        # Average throughput across all grid sizes for each block size
        avg_throughput = df_block.groupby('block_size')['throughput_mcells_s'].mean()
        block_sizes = sorted(df_block['block_size'].unique())
        
        bars = ax3.bar(range(len(block_sizes)), 
                      [avg_throughput[bs] for bs in block_sizes],
                      color=COLORS['cuda'], alpha=0.7, edgecolor='black', linewidth=1)
        
        # Highlight optimal
        optimal_idx = avg_throughput.idxmax()
        optimal_pos = block_sizes.index(optimal_idx)
        bars[optimal_pos].set_color(COLORS['optimal'])
        bars[optimal_pos].set_alpha(0.9)
        
        ax3.set_xlabel('Block Size (threads/dim)')
        ax3.set_ylabel('Avg. Throughput (M cells/s)')
        ax3.set_title('Block Size Impact', fontweight='bold')
        ax3.set_xticks(range(len(block_sizes)))
        ax3.set_xticklabels([f'{bs}×{bs}' for bs in block_sizes])
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add optimal marker
        ax3.text(optimal_pos, avg_throughput[optimal_idx] * 1.02, 
                'Optimal', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        fig3.tight_layout()
        output_path = output_dir / '3_block_size_optimization.png'
        fig3.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        plt.close(fig3)
    
    # Plot 4: Parallel Efficiency
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    if 'efficiencies' in metrics and 'common_sizes' in metrics:
        sizes = metrics['common_sizes']
        efficiencies = metrics['efficiencies']
        
        ax4.bar(range(len(sizes)), efficiencies, 
               color=COLORS['cuda'], alpha=0.7, edgecolor='black', linewidth=1)
        ax4.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Ideal (100%)')
        
        ax4.set_xlabel('Grid Size (N×N)')
        ax4.set_ylabel('Parallel Efficiency (%)')
        ax4.set_title('GPU Utilization Efficiency', fontweight='bold')
        ax4.set_xticks(range(len(sizes)))
        ax4.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=45, ha='right')
        ax4.legend(frameon=True, fancybox=False, edgecolor='black')
        ax4.grid(True, axis='y', alpha=0.3)
        
        fig4.tight_layout()
        output_path = output_dir / '4_parallel_efficiency.png'
        fig4.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        plt.close(fig4)
    
    # Plot 5: Time per Generation (Scaling)
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    if df_seq is not None and df_cuda is not None:
        sizes = sorted(set(df_seq['grid_size'].unique()) & set(df_cuda['grid_size'].unique()))
        
        python_time = [df_seq[df_seq['grid_size'] == s]['time_per_gen_ms'].values[0] for s in sizes]
        cuda_time = [df_cuda[df_cuda['grid_size'] == s]['time_per_gen_ms'].values[0] for s in sizes]
        
        width = 0.35
        x = np.arange(len(sizes))
        
        ax5.bar(x - width/2, python_time, width, label='Python (CPU)',
               color=COLORS['python'], alpha=0.8, edgecolor='black', linewidth=1)
        ax5.bar(x + width/2, cuda_time, width, label='CUDA (GPU)',
               color=COLORS['cuda'], alpha=0.8, edgecolor='black', linewidth=1)
        
        ax5.set_xlabel('Grid Size (N×N)')
        ax5.set_ylabel('Time per Generation (ms)')
        ax5.set_title('Execution Time Comparison', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=45, ha='right')
        ax5.set_yscale('log')
        ax5.legend(frameon=True, fancybox=False, edgecolor='black')
        ax5.grid(True, which='both', alpha=0.3, axis='y')
        
        fig5.tight_layout()
        output_path = output_dir / '5_execution_time.png'
        fig5.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_files.append(output_path)
        plt.close(fig5)
    
    return saved_files

def create_block_size_dashboard(df, df_seq, output_dir):
    """Academic-style block size analysis - save individual plots"""
    saved_files = []
    
    grid_sizes = sorted(df['grid_size'].unique())
    block_sizes = sorted(df['block_size'].unique())
    
    # Use colorblind-friendly palette
    colors = plt.colormaps['tab10'](np.linspace(0, 0.9, len(grid_sizes)))
    
    # Plot 1: Throughput vs Block Size
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for idx, grid in enumerate(grid_sizes):
        data = df[df['grid_size'] == grid].sort_values('block_size')
        
        ax1.plot(data['block_size'], data['throughput_mcells_s'], 
                'o-', color=colors[idx], linewidth=2, markersize=6,
                label=f'{grid}×{grid}', markeredgecolor='white', markeredgewidth=0.5)
        
        # Mark maximum
        max_idx = data['throughput_mcells_s'].idxmax()
        max_point = data.loc[max_idx]
        ax1.plot(max_point['block_size'], max_point['throughput_mcells_s'],
                '*', color=colors[idx], markersize=15, markeredgecolor='black',
                markeredgewidth=1, zorder=10)
    
    # Add Python baseline if available
    if df_seq is not None:
        for idx, grid in enumerate(grid_sizes):
            seq_data = df_seq[df_seq['grid_size'] == grid]
            if not seq_data.empty:
                seq_tp = seq_data['throughput_mcells_s'].values[0]
                ax1.axhline(y=seq_tp, color=colors[idx], linestyle=':', 
                           linewidth=1.5, alpha=0.5)
    
    ax1.set_xlabel('Block Size (threads per dimension)')
    ax1.set_ylabel('Throughput (M cells/s)')
    ax1.set_title('Throughput vs Block Size', fontweight='bold')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(block_sizes)
    ax1.set_xticklabels([f'{bs}×{bs}' for bs in block_sizes])
    
    fig1.tight_layout()
    output_path = output_dir / '6_throughput_vs_blocksize.png'
    fig1.savefig(output_path, dpi=300, bbox_inches='tight')
    saved_files.append(output_path)
    plt.close(fig1)
    
    # Plot 2: Heatmap
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    pivot = df.pivot(index='grid_size', columns='block_size', values='throughput_mcells_s')
    
    im = ax2.imshow(pivot.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    ax2.set_xticks(np.arange(len(block_sizes)))
    ax2.set_yticks(np.arange(len(grid_sizes)))
    ax2.set_xticklabels([f'{bs}×{bs}' for bs in block_sizes])
    ax2.set_yticklabels([f'{gs}×{gs}' for gs in grid_sizes])
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Grid Size')
    ax2.set_title('Throughput Heatmap (M cells/s)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(grid_sizes)):
        for j in range(len(block_sizes)):
            value = pivot.values[i, j]
            text_color = 'white' if value > pivot.values.max() * 0.6 else 'black'
            ax2.text(j, i, f'{value:.0f}', ha="center", va="center", 
                    color=text_color, fontsize=9, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Throughput', rotation=270, labelpad=15)
    
    fig2.tight_layout()
    output_path = output_dir / '7_throughput_heatmap.png'
    fig2.savefig(output_path, dpi=300, bbox_inches='tight')
    saved_files.append(output_path)
    plt.close(fig2)
    
    # Plot 3: Speedup Relative to Baseline
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    
    speedup_data = []
    baseline_bs = min(block_sizes)
    
    for bs in block_sizes:
        speedups = []
        for grid in grid_sizes:
            baseline_time = df[(df['grid_size'] == grid) & 
                              (df['block_size'] == baseline_bs)]['time_ms'].values[0]
            current_time = df[(df['grid_size'] == grid) & 
                             (df['block_size'] == bs)]['time_ms'].values[0]
            speedups.append(baseline_time / current_time)
        speedup_data.append(np.mean(speedups))
    
    bars = ax3.barh(range(len(block_sizes)), speedup_data,
                   color=COLORS['cuda'], alpha=0.7, edgecolor='black', linewidth=1)
    
    # Highlight best
    max_idx = np.argmax(speedup_data)
    bars[max_idx].set_color(COLORS['optimal'])
    bars[max_idx].set_alpha(0.9)
    
    ax3.set_yticks(range(len(block_sizes)))
    ax3.set_yticklabels([f'{bs}×{bs}' for bs in block_sizes])
    ax3.set_xlabel('Average Speedup')
    ax3.set_title(f'Speedup vs BS={baseline_bs}', fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)
    ax3.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add values
    for i, speedup in enumerate(speedup_data):
        ax3.text(speedup + 0.05, i, f'{speedup:.2f}×', 
                va='center', fontsize=9, fontweight='bold')
    
    fig3.tight_layout()
    output_path = output_dir / '8_blocksize_speedup.png'
    fig3.savefig(output_path, dpi=300, bbox_inches='tight')
    saved_files.append(output_path)
    plt.close(fig3)
    
    return saved_files


def print_summary(data, metrics):
    """Print comprehensive terminal summary"""
    print("\n" + "="*90)
    print("║" + " "*88 + "║")
    print("║" + " "*25 + "PERFORMANCE SUMMARY" + " "*33 + "║")
    print("║" + " "*88 + "║")
    print("="*90)
    
    # Block size results
    if 'block_sizes' in data:
        df = data['block_sizes']
        best = df.loc[df['throughput_mcells_s'].idxmax()]
        print(f"\nOPTIMAL BLOCK SIZE CONFIGURATION:")
        print(f"   Block Size: {int(best['block_size'])}×{int(best['block_size'])} ({int(best['block_size']**2)} threads/block)")
        print(f"   Peak Throughput: {best['throughput_mcells_s']:.2f} M cells/s")
        print(f"   Grid Size: {int(best['grid_size'])}×{int(best['grid_size'])}")
    
    # Speedup statistics
    if 'speedups' in metrics and len(metrics['speedups']) > 0:
        avg_speedup = np.mean(metrics['speedups'])
        max_speedup = np.max(metrics['speedups'])
        min_speedup = np.min(metrics['speedups'])
        
        print(f"\nSPEEDUP STATISTICS (CUDA vs Python):")
        print(f"   Average Speedup: {avg_speedup:>8.2f}×")
        print(f"   Maximum Speedup: {max_speedup:>8.2f}×")
        print(f"   Minimum Speedup: {min_speedup:>8.2f}×")
    
    if metrics.get('breakeven_size'):
        print(f"\nBREAK-EVEN POINT:")
        print(f"   Grid Size: {metrics['breakeven_size']}×{metrics['breakeven_size']} (CUDA starts beating Python)")
    
    if 'efficiencies' in metrics and len(metrics['efficiencies']) > 0:
        avg_efficiency = np.mean(metrics['efficiencies'])
        max_efficiency = np.max(metrics['efficiencies'])
        
        print(f"\nPARALLEL EFFICIENCY:")
        print(f"   Average Efficiency: {avg_efficiency:>7.2f}%")
        print(f"   Maximum Efficiency: {max_efficiency:>7.2f}%")
    
    if 'strong_scaling' in metrics:
        data = metrics['strong_scaling']
        best_bs = data.loc[data['speedup'].idxmax()]
        
        print(f"\nSTRONG SCALING (Fixed Grid):")
        print(f"   Best Block Size: {int(best_bs['block_size'])}×{int(best_bs['block_size'])}")
        print(f"   Speedup vs BS=1: {best_bs['speedup']:>8.2f}×")
        print(f"   Efficiency: {best_bs['efficiency']:>7.2f}%")
    
    print("\n" + "="*90 + "\n")

def main():
    """Main analysis function"""
    print("\n" + "="*70)
    print("  Game of Life - Academic Performance Analysis")
    print("  Publication-Quality Plots")
    print("="*70)
    
    print("\nLoading benchmark data...")
    data = load_data()
    
    if not data:
        print("\nError: No benchmark data found!")
        print("Please run benchmarks first: ./run.sh benchmark")
        sys.exit(1)
    
    print("Calculating performance metrics...")
    metrics = calculate_metrics(
        data.get('sequential'),
        data.get('cuda'),
        data.get('block_sizes')
    )
    
    print("\nGenerating academic-style visualizations...\n")
    print("  Each plot saved as individual PNG file (300 DPI)\n")
    
    output_dir = Path('benchmarks')
    output_dir.mkdir(exist_ok=True)
    
    all_saved_files = []
    
    # Create main academic plots
    if data.get('sequential') is not None and data.get('cuda') is not None:
        print("  Creating performance analysis plots...")
        saved_files = create_academic_plots(
            data.get('sequential'),
            data.get('cuda'),
            data.get('block_sizes'),
            metrics,
            output_dir
        )
        all_saved_files.extend(saved_files)
        for path in saved_files:
            print(f"    ✓ {path.name}")
    
    # Create block size analysis
    if 'block_sizes' in data:
        print("\n  Creating block size analysis plots...")
        saved_files = create_block_size_dashboard(
            data['block_sizes'],
            data.get('sequential'),
            output_dir
        )
        all_saved_files.extend(saved_files)
        for path in saved_files:
            print(f"    ✓ {path.name}")
    
    # Print terminal summary
    print()
    print_summary(data, metrics)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print(f"\n{len(all_saved_files)} individual plots generated:")
    for path in all_saved_files:
        print(f"   📊 {path.name}")
    
    print("\nPlot details:")
    print("   • Resolution: 300 DPI")
    print("   • Format: PNG")
    print("   • Style: Academic publication quality")
    print("   • Fonts: Serif (Times New Roman)")
    print("   • Colors: Colorblind-friendly palette")
    print("="*70)

if __name__ == "__main__":
    main()
