#!/usr/bin/env python3
"""
Complete Performance Analysis & Visualization Suite
Comprehensive analysis of CUDA block sizes, CUDA vs CPU speedup, scaling, and efficiency
Merges functionality from visualize_block_sizes.py and analyze_performance.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cm
import numpy as np
import sys
from pathlib import Path
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = {
    'primary': '#2E86AB',    # Blu oceano
    'secondary': '#A23B72',  # Magenta
    'success': '#06A77D',    # Verde acqua
    'warning': '#F18F01',    # Arancione
    'danger': '#C73E1D',     # Rosso
    'gradient': ['#667eea', '#764ba2', '#f093fb', '#4facfe']
}

TITLE_FONT = {'family': 'sans-serif', 'weight': 'bold', 'size': 16}
LABEL_FONT = {'family': 'sans-serif', 'weight': 'normal', 'size': 12}
LEGEND_FONT = FontProperties(family='sans-serif', size=10)

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

def create_block_size_dashboard(df, df_seq=None):
    """Dashboard for block size performance analysis"""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Titolo principale
    fig.suptitle('Game of Life: CUDA Block Size Performance Analysis', 
                 fontsize=22, fontweight='bold', 
                 color=COLORS['primary'], y=0.98)
    
    grid_sizes = sorted(df['grid_size'].unique())
    block_sizes = sorted(df['block_size'].unique())
    
    # PANEL 1: Throughput Lines
    ax1 = fig.add_subplot(gs[0, :2])
    
    colors_gradient = cm.get_cmap('viridis')(np.linspace(0.2, 0.9, len(grid_sizes)))
    
    # Check if we have error data
    has_error_bars = 'std_throughput_mcells_s' in df.columns
    
    for idx, grid in enumerate(grid_sizes):
        data = df[df['grid_size'] == grid]
        
        # Plot with error bars if available
        if has_error_bars and not data['std_throughput_mcells_s'].isna().all():
            ax1.errorbar(data['block_size'], data['throughput_mcells_s'],
                        yerr=data['std_throughput_mcells_s'],
                        marker='o', linewidth=3, markersize=10,
                        label=f'{grid}×{grid} (CUDA)',
                        color=colors_gradient[idx],
                        markeredgecolor='white', markeredgewidth=2,
                        capsize=5, capthick=2, elinewidth=2, alpha=0.9)
        else:
            ax1.plot(data['block_size'], data['throughput_mcells_s'], 
                    marker='o', linewidth=3, markersize=10, 
                    label=f'{grid}×{grid} (CUDA)', color=colors_gradient[idx],
                    markeredgecolor='white', markeredgewidth=2)
        
        # Evidenzia il massimo
        max_idx = data['throughput_mcells_s'].idxmax()
        max_point = data.loc[max_idx]
        ax1.scatter(max_point['block_size'], max_point['throughput_mcells_s'],
                   s=300, marker='*', color='gold', edgecolor='black', 
                   linewidth=2, zorder=10)
    
    # Aggiungi linee sequenziali se disponibili
    if df_seq is not None:
        for idx, grid in enumerate(grid_sizes):
            seq_data = df_seq[df_seq['grid_size'] == grid]
            if not seq_data.empty:
                seq_throughput = seq_data['throughput_mcells_s'].values[0]
                ax1.axhline(y=seq_throughput, color=colors_gradient[idx], 
                           linestyle='--', linewidth=2, alpha=0.5,
                           label=f'{grid}×{grid} (CPU)')
    
    ax1.set_xlabel('Block Size (threads per dimension)', **LABEL_FONT)
    ax1.set_ylabel('Throughput (M cells/s)', **LABEL_FONT)
    
    # Add note about error bars if present
    title_text = 'Performance Throughput (CUDA vs CPU)'
    if has_error_bars:
        title_text += ' [with ±1σ error bars]'
    ax1.set_title(title_text, **TITLE_FONT, pad=15)
    
    ax1.legend(loc='upper left', framealpha=0.95, prop=LEGEND_FONT, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(block_sizes)
    ax1.set_xticklabels([f'{bs}×{bs}' for bs in block_sizes])
    ax1.axvspan(12, 20, alpha=0.1, color='green', label='Optimal Zone')
    ax1.set_facecolor('#f8f9fa')
    
    # PANEL 2: Winner Badge
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    best = df.loc[df['throughput_mcells_s'].idxmax()]
    best_bs = int(best['block_size'])
    
    from matplotlib.patches import Circle
    circle = Circle((0.5, 0.6), 0.35, color=COLORS['success'], alpha=0.2)
    ax2.add_patch(circle)
    
    ax2.text(0.5, 0.85, 'WINNER', ha='center', va='center',
            fontsize=18, fontweight='bold', color=COLORS['success'])
    ax2.text(0.5, 0.6, f'{best_bs}×{best_bs}', ha='center', va='center',
            fontsize=48, fontweight='bold', color=COLORS['primary'])
    ax2.text(0.5, 0.35, f'{int(best_bs**2)} threads/block', ha='center', va='center',
            fontsize=12, style='italic', color='gray')
    ax2.text(0.5, 0.15, f'{best["throughput_mcells_s"]:.1f} M cells/s', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=COLORS['success'])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # PANEL 3: Heatmap
    ax3 = fig.add_subplot(gs[1, :2])
    
    pivot = df.pivot(index='grid_size', columns='block_size', values='throughput_mcells_s')
    
    im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                    interpolation='nearest', vmin=pivot.values.min()*0.8)
    
    ax3.set_xticks(np.arange(len(block_sizes)))
    ax3.set_yticks(np.arange(len(grid_sizes)))
    ax3.set_xticklabels([f'{bs}×{bs}' for bs in block_sizes])
    ax3.set_yticklabels([f'{gs}×{gs}' for gs in grid_sizes])
    ax3.set_xlabel('Block Size', **LABEL_FONT)
    ax3.set_ylabel('Grid Size', **LABEL_FONT)
    ax3.set_title('Performance Heatmap (M cells/s)', **TITLE_FONT, pad=15)
    
    # Aggiungi valori
    for i in range(len(grid_sizes)):
        for j in range(len(block_sizes)):
            value = pivot.values[i, j]
            text_color = 'white' if value < pivot.values.max() * 0.6 else 'black'
            text = ax3.text(j, i, f'{value:.0f}',
                          ha="center", va="center", 
                          color=text_color, fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Throughput', rotation=270, labelpad=20, **LABEL_FONT)
    
    # PANEL 4: Speedup Comparison
    ax4 = fig.add_subplot(gs[1, 2])
    
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
    
    bars = ax4.barh(range(len(block_sizes)), speedup_data, 
                    color=[COLORS['danger'], COLORS['warning'], 
                           COLORS['success'], COLORS['secondary']], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    max_idx = np.argmax(speedup_data)
    bars[max_idx].set_color(COLORS['success'])
    bars[max_idx].set_alpha(1.0)
    bars[max_idx].set_edgecolor('gold')
    bars[max_idx].set_linewidth(3)
    
    ax4.set_yticks(range(len(block_sizes)))
    ax4.set_yticklabels([f'{bs}×{bs}' for bs in block_sizes])
    ax4.set_xlabel('Average Speedup', **LABEL_FONT)
    ax4.set_title(f'Speedup vs BS={baseline_bs}', **TITLE_FONT, pad=15)
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax4.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    for i, (bs, speedup) in enumerate(zip(block_sizes, speedup_data)):
        ax4.text(speedup + 0.5, i, f'{speedup:.1f}×', 
                va='center', fontweight='bold', fontsize=11)
    
    ax4.set_facecolor('#f8f9fa')
    
    # PANEL 5: Execution Time
    ax5 = fig.add_subplot(gs[2, :2])
    
    width = 0.15
    x = np.arange(len(grid_sizes))
    
    for idx, bs in enumerate(block_sizes):
        data = [df[(df['grid_size'] == g) & (df['block_size'] == bs)]['time_per_gen_ms'].values[0] 
                for g in grid_sizes]
        offset = (idx - len(block_sizes)/2 + 0.5) * width
        bars = ax5.bar(x + offset, data, width, 
                      label=f'BS={bs}×{bs} (CUDA)',
                      alpha=0.85, edgecolor='white', linewidth=1)
    
    if df_seq is not None:
        seq_data = [df_seq[df_seq['grid_size'] == g]['time_per_gen_ms'].values[0] 
                    if not df_seq[df_seq['grid_size'] == g].empty else np.nan
                    for g in grid_sizes]
        offset = (len(block_sizes) - len(block_sizes)/2 + 0.5) * width
        ax5.bar(x + offset, seq_data, width, 
               label='Sequential (CPU)',
               alpha=0.85, edgecolor='black', linewidth=2,
               color='red', hatch='//')
    
    ax5.set_xlabel('Grid Size', **LABEL_FONT)
    ax5.set_ylabel('Time per Generation (ms, log scale)', **LABEL_FONT)
    ax5.set_title('⏱️ Execution Time: CUDA vs CPU', **TITLE_FONT, pad=15)
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{gs}×{gs}' for gs in grid_sizes], rotation=45)
    ax5.legend(loc='upper left', ncol=3, prop=LEGEND_FONT, framealpha=0.95, fontsize=8)
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, which='both', linestyle='--')
    ax5.set_facecolor('#f8f9fa')
    
    # PANEL 6: Occupancy Theory
    ax6 = fig.add_subplot(gs[2, 2])
    
    MAX_THREADS_PER_SM = 2048
    occupancy_data = [(bs**2 / MAX_THREADS_PER_SM) * 
                      min(MAX_THREADS_PER_SM // (bs**2), 32) * 100 
                      for bs in block_sizes]
    
    colors_occ = [COLORS['danger'] if o < 50 else 
                  COLORS['warning'] if o < 75 else 
                  COLORS['success'] for o in occupancy_data]
    
    pie_result = ax6.pie(occupancy_data, 
                        labels=[f'BS={bs}×{bs}' for bs in block_sizes],
                        colors=colors_occ, autopct='%1.0f%%',
                        startangle=90, pctdistance=0.85,
                        explode=[0.05 if i == max_idx else 0 
                                for i in range(len(block_sizes))],
                        shadow=True)
    
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
    else:
        wedges, texts = pie_result
        autotexts = []
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax6.set_title('Theoretical Occupancy', **TITLE_FONT, pad=15)
    
    # Footer
    footer_text = (f"Generated from: {len(df)} benchmark runs | "
                  f"Grid sizes: {min(grid_sizes)}–{max(grid_sizes)} | "
                  f"Block sizes: {min(block_sizes)}–{max(block_sizes)} | "
                  f"Best: {best_bs}×{best_bs} threads/block")
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=9, 
            style='italic', color='gray')
    
    return fig

def create_speedup_dashboard(df_seq, df_cuda, df_block, metrics):
    """Comprehensive speedup analysis dashboard"""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Performance Analysis: Speedup, Scaling & Efficiency', 
                 fontsize=22, fontweight='bold', 
                 color=COLORS['primary'], y=0.98)
    
    # PANEL 1: CUDA vs Python Speedup
    ax1 = fig.add_subplot(gs[0, :2])
    
    if 'common_sizes' in metrics and 'speedups' in metrics:
        sizes = metrics['common_sizes']
        speedups = metrics['speedups']
        
        ax1.plot(sizes, speedups, marker='o', linewidth=3, markersize=12,
                color=COLORS['success'], markeredgecolor='white', 
                markeredgewidth=2, label='CUDA Speedup vs Python')
        
        ideal_speedup = [s / sizes[0] for s in sizes]
        ax1.plot(sizes, ideal_speedup, linestyle='--', linewidth=2, 
                color='gray', alpha=0.5, label='Ideal Linear Scaling')
        
        if metrics.get('breakeven_size'):
            breakeven_size = metrics['breakeven_size']
            ax1.axvline(x=breakeven_size, color=COLORS['warning'], 
                       linestyle='--', linewidth=2, alpha=0.7)
            ax1.text(breakeven_size, max(speedups) * 0.9, 
                    f'Break-even\n{breakeven_size}×{breakeven_size}',
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=COLORS['warning'], alpha=0.3))
        
        for size, speedup in zip(sizes, speedups):
            ax1.text(size, speedup, f'{speedup:.1f}×', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Grid Size (N×N)', **LABEL_FONT)
        ax1.set_ylabel('Speedup Factor (×)', **LABEL_FONT)
        ax1.set_title('CUDA Speedup vs Python Sequential', **TITLE_FONT, pad=15)
        ax1.legend(loc='upper left', prop=LEGEND_FONT, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xscale('log', base=2)
        ax1.set_facecolor('#f8f9fa')
        ax1.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # PANEL 2: Best Config Badge
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    if df_block is not None:
        best = df_block.loc[df_block['throughput_mcells_s'].idxmax()]
        avg_throughput = df_block.groupby('block_size')['throughput_mcells_s'].mean()
        best_avg_bs = avg_throughput.idxmax()
        
        ax2.text(0.5, 0.9, 'BEST CONFIG', ha='center', va='center',
                fontsize=14, fontweight='bold', color=COLORS['success'])
        ax2.text(0.5, 0.7, f"BS={int(best['block_size'])}×{int(best['block_size'])}", 
                ha='center', va='center', fontsize=32, fontweight='bold', 
                color=COLORS['primary'])
        ax2.text(0.5, 0.5, f"{best['throughput_mcells_s']:.1f} M cells/s", 
                ha='center', va='center', fontsize=14, color=COLORS['success'])
        ax2.text(0.5, 0.3, f"Grid: {int(best['grid_size'])}×{int(best['grid_size'])}", 
                ha='center', va='center', fontsize=12, style='italic', color='gray')
        ax2.text(0.5, 0.1, f"Avg Best BS: {int(best_avg_bs)}×{int(best_avg_bs)}", 
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    # PANEL 3: Python vs CUDA Best vs CUDA Worst
    ax3 = fig.add_subplot(gs[1, :])
    
    if df_seq is not None and df_block is not None:
        sizes = sorted(set(df_seq['grid_size'].unique()) & set(df_block['grid_size'].unique()))
        
        best_bs = 16
        worst_bs = 1
        
        python_throughput = [df_seq[df_seq['grid_size'] == s]['throughput_mcells_s'].values[0] 
                            for s in sizes]
        cuda_best = [df_block[(df_block['grid_size'] == s) & 
                             (df_block['block_size'] == best_bs)]['throughput_mcells_s'].values[0]
                    if len(df_block[(df_block['grid_size'] == s) & 
                                   (df_block['block_size'] == best_bs)]) > 0 else 0
                    for s in sizes]
        cuda_worst = [df_block[(df_block['grid_size'] == s) & 
                              (df_block['block_size'] == worst_bs)]['throughput_mcells_s'].values[0]
                     if len(df_block[(df_block['grid_size'] == s) & 
                                    (df_block['block_size'] == worst_bs)]) > 0 else 0
                     for s in sizes]
        
        x = np.arange(len(sizes))
        width = 0.25
        
        ax3.bar(x - width, python_throughput, width, 
               label='Python Sequential (CPU)', 
               color=COLORS['danger'], alpha=0.8, edgecolor='black', linewidth=2)
        ax3.bar(x, cuda_best, width, 
               label=f'CUDA Best (BS={best_bs}×{best_bs})', 
               color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=2)
        ax3.bar(x + width, cuda_worst, width, 
               label=f'CUDA Worst (BS={worst_bs}×{worst_bs})', 
               color=COLORS['warning'], alpha=0.8, edgecolor='black', linewidth=2, hatch='//')
        
        ax3.set_xlabel('Grid Size (N×N)', **LABEL_FONT)
        ax3.set_ylabel('Throughput (M cells/s)', **LABEL_FONT)
        ax3.set_title('Throughput: Python vs CUDA (Best vs Worst)', **TITLE_FONT, pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=45)
        ax3.legend(prop=LEGEND_FONT, framealpha=0.95, loc='upper left')
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_facecolor('#f8f9fa')
        ax3.set_yscale('log')
    
    # PANEL 4: Strong Scaling
    ax4 = fig.add_subplot(gs[2, 0])
    
    if 'strong_scaling' in metrics:
        data = metrics['strong_scaling']
        
        ax4.plot(data['block_size'], data['speedup'], marker='o', linewidth=3,
                markersize=10, color=COLORS['primary'], label='Actual Speedup',
                markeredgecolor='white', markeredgewidth=2)
        ax4.plot(data['block_size'], data['ideal_speedup'], linestyle='--', 
                linewidth=2, color='gray', alpha=0.5, label='Ideal Speedup')
        
        ax4.set_xlabel('Block Size (threads/dim)', **LABEL_FONT)
        ax4.set_ylabel('Speedup vs BS=1', **LABEL_FONT)
        ax4.set_title('Strong Scaling', **TITLE_FONT, pad=15)
        ax4.legend(prop=LEGEND_FONT, framealpha=0.95)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_facecolor('#f8f9fa')
        ax4.set_yscale('log')
        ax4.set_xscale('log', base=2)
    
    # PANEL 5: Parallel Efficiency
    ax5 = fig.add_subplot(gs[2, 1])
    
    if 'efficiencies' in metrics:
        sizes = metrics['common_sizes']
        efficiencies = metrics['efficiencies']
        
        colors = [COLORS['success'] if e > 2 else 
                 COLORS['warning'] if e > 1 else 
                 COLORS['danger'] for e in efficiencies]
        
        bars = ax5.bar(range(len(sizes)), efficiencies, color=colors, 
                      alpha=0.8, edgecolor='black', linewidth=2)
        
        for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax5.set_xlabel('Grid Size (N×N)', **LABEL_FONT)
        ax5.set_ylabel('Parallel Efficiency (%)', **LABEL_FONT)
        ax5.set_title('GPU Utilization Efficiency', **TITLE_FONT, pad=15)
        ax5.set_xticks(range(len(sizes)))
        ax5.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=45)
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax5.set_facecolor('#f8f9fa')
        ax5.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # PANEL 6: Weak Scaling
    ax6 = fig.add_subplot(gs[2, 2])
    
    if 'weak_scaling' in metrics:
        colors_gradient = cm.get_cmap('viridis')(np.linspace(0.2, 0.9, len(metrics['weak_scaling'])))
        
        for idx, item in enumerate(metrics['weak_scaling']):
            bs = item['block_size']
            data = item['data']
            ax6.plot(data['grid_size'], data['time_per_gen_ms'], 
                    marker='o', linewidth=2, markersize=8,
                    label=f'BS={bs}×{bs}', color=colors_gradient[idx],
                    markeredgecolor='white', markeredgewidth=1.5)
        
        ax6.set_xlabel('Grid Size (N×N)', **LABEL_FONT)
        ax6.set_ylabel('Time per Generation (ms)', **LABEL_FONT)
        ax6.set_title('Weak Scaling', **TITLE_FONT, pad=15)
        ax6.legend(prop=LEGEND_FONT, framealpha=0.95, fontsize=8)
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.set_facecolor('#f8f9fa')
        ax6.set_yscale('log')
        ax6.set_xscale('log', base=2)
    
    # Footer
    if 'speedups' in metrics and len(metrics['speedups']) > 0:
        avg_speedup = np.mean(metrics['speedups'])
        max_speedup = np.max(metrics['speedups'])
        footer = (f"Avg Speedup: {avg_speedup:.1f}× | "
                 f"Max Speedup: {max_speedup:.1f}× | "
                 f"Break-even: {metrics.get('breakeven_size', 'N/A')}×{metrics.get('breakeven_size', 'N/A')}")
        fig.text(0.5, 0.02, footer, ha='center', fontsize=10, 
                fontweight='bold', color=COLORS['primary'])
    
    return fig

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
    print("\033[0;36m")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║      Game of Life - Performance Analysis                   ║")
    print("║                                                            ║")
    print("║   Block Sizes | Speedup | Scaling | Comprehensive Stats   ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\033[0m")
    
    print("\nLoading benchmark data...")
    data = load_data()
    
    if not data:
        print("\n\033[0;31mError: No benchmark data found!\033[0m")
        print("\nPlease run benchmarks first:")
        print("  ./run.sh benchmark")
        sys.exit(1)
    
    print("\nCalculating performance metrics...")
    metrics = calculate_metrics(
        data.get('sequential'),
        data.get('cuda'),
        data.get('block_sizes')
    )
    
    print("\nGenerating analysis visualizations...\n")
    
    output_dir = Path('benchmarks')
    output_dir.mkdir(exist_ok=True)
    
    # Create block size dashboard if data available
    if 'block_sizes' in data:
        print("  Creating block size dashboard...")
        fig1 = create_block_size_dashboard(data['block_sizes'], data.get('sequential'))
        output_path = output_dir / 'block_size_dashboard.png'
        fig1.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {output_path}")
    
    # Create speedup dashboard if data available
    if data.get('sequential') is not None and data.get('cuda') is not None:
        print("  Creating speedup analysis dashboard...")
        fig2 = create_speedup_dashboard(
            data.get('sequential'),
            data.get('cuda'),
            data.get('block_sizes'),
            metrics
        )
        output_path = output_dir / 'performance_analysis_dashboard.png'
        fig2.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    Saved: {output_path}")
    
    # Print terminal summary
    print_summary(data, metrics)
    
    print("Analysis complete!")
    print("\nOutput files:")
    if 'block_sizes' in data:
        print("   - block_size_dashboard.png           (Block size analysis)")
    if data.get('sequential') is not None and data.get('cuda') is not None:
        print("   - performance_analysis_dashboard.png (Speedup, scaling, efficiency)")
    
    print("\nShowing interactive plots...")
    plt.show()

if __name__ == "__main__":
    main()
