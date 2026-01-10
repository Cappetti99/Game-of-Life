#!/usr/bin/env python3
"""
Visualizzatore risultati benchmark block size - VERSIONE PROFESSIONALE
Crea grafici estetici per analizzare l'impatto del numero di thread
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
import numpy as np
import sys
from pathlib import Path
import seaborn as sns

# Configurazione estetica globale
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Colori personalizzati (palette moderna)
COLORS = {
    'primary': '#2E86AB',    # Blu oceano
    'secondary': '#A23B72',  # Magenta
    'success': '#06A77D',    # Verde acqua
    'warning': '#F18F01',    # Arancione
    'danger': '#C73E1D',     # Rosso
    'gradient': ['#667eea', '#764ba2', '#f093fb', '#4facfe']
}

# Font settings
TITLE_FONT = {'family': 'sans-serif', 'weight': 'bold', 'size': 16}
LABEL_FONT = {'family': 'sans-serif', 'weight': 'normal', 'size': 12}
LEGEND_FONT = FontProperties(family='sans-serif', size=10)

def load_results(csv_path):
    """Carica i risultati dal CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} results from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {csv_path}")
        print("Run the benchmark script first!")
        sys.exit(1)

def load_sequential_results(csv_path):
    """Carica i risultati sequenziali dal CSV"""
    try:
        df = pd.read_csv(csv_path)
        # Rinomina colonne per compatibilità
        df = df.rename(columns={
            'size': 'grid_size',
            'total_time_ms': 'time_ms',
            'time_per_generation_ms': 'time_per_gen_ms',
            'cells_per_second_million': 'throughput_mcells_s'
        })
        print(f"✓ Loaded {len(df)} sequential results from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"⚠ Sequential benchmark not found: {csv_path}")
        return None

def create_main_dashboard(df, df_seq=None):
    """Dashboard principale con 6 pannelli"""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Titolo principale con stile
    fig.suptitle('Game of Life: CUDA Block Size Performance Analysis', 
                 fontsize=22, fontweight='bold', 
                 color=COLORS['primary'], y=0.98)
    
    grid_sizes = sorted(df['grid_size'].unique())
    block_sizes = sorted(df['block_size'].unique())
    
    # ========================================================================
    # PANEL 1: Throughput Lines (grande, 2 colonne)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    import matplotlib.cm as cm
    colors_gradient = cm.get_cmap('viridis')(np.linspace(0.2, 0.9, len(grid_sizes)))
    
    for idx, grid in enumerate(grid_sizes):
        data = df[df['grid_size'] == grid]
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
    ax1.set_title('🚀 Performance Throughput (CUDA vs CPU)', **TITLE_FONT, pad=15)
    ax1.legend(loc='upper left', framealpha=0.95, prop=LEGEND_FONT, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(block_sizes)
    ax1.set_xticklabels([f'{bs}×{bs}' for bs in block_sizes])
    
    # Aggiungi banda per "zona ottimale"
    ax1.axvspan(12, 20, alpha=0.1, color='green', label='Optimal Zone')
    ax1.set_facecolor('#f8f9fa')
    
    # ========================================================================
    # PANEL 2: Winner Badge
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Trova il vincitore assoluto
    best = df.loc[df['throughput_mcells_s'].idxmax()]
    best_bs = int(best['block_size'])
    
    # Crea un "badge" visivo
    from matplotlib.patches import Circle
    circle = Circle((0.5, 0.6), 0.35, color=COLORS['success'], alpha=0.2)
    ax2.add_patch(circle)
    
    ax2.text(0.5, 0.85, '🏆 WINNER', ha='center', va='center',
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
    
    # ========================================================================
    # PANEL 3: Heatmap Interattiva
    # ========================================================================
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
    ax3.set_title('🔥 Performance Heatmap (M cells/s)', **TITLE_FONT, pad=15)
    
    # Aggiungi valori con colori adattivi
    for i in range(len(grid_sizes)):
        for j in range(len(block_sizes)):
            value = pivot.values[i, j]
            # Colore testo basato su luminosità sfondo
            text_color = 'white' if value < pivot.values.max() * 0.6 else 'black'
            text = ax3.text(j, i, f'{value:.0f}',
                          ha="center", va="center", 
                          color=text_color, fontsize=11, fontweight='bold')
    
    # Colorbar con stile
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Throughput', rotation=270, labelpad=20, **LABEL_FONT)
    
    # ========================================================================
    # PANEL 4: Speedup Comparison
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Calcola speedup medio per ogni block size
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
    
    # Evidenzia il migliore
    max_idx = np.argmax(speedup_data)
    bars[max_idx].set_color(COLORS['success'])
    bars[max_idx].set_alpha(1.0)
    bars[max_idx].set_edgecolor('gold')
    bars[max_idx].set_linewidth(3)
    
    ax4.set_yticks(range(len(block_sizes)))
    ax4.set_yticklabels([f'{bs}×{bs}' for bs in block_sizes])
    ax4.set_xlabel('Average Speedup', **LABEL_FONT)
    ax4.set_title(f'⚡ Speedup vs BS={baseline_bs}', **TITLE_FONT, pad=15)
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax4.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Aggiungi valori sulle barre
    for i, (bs, speedup) in enumerate(zip(block_sizes, speedup_data)):
        ax4.text(speedup + 0.5, i, f'{speedup:.1f}×', 
                va='center', fontweight='bold', fontsize=11)
    
    ax4.set_facecolor('#f8f9fa')
    
    # ========================================================================
    # PANEL 5: Execution Time (Log scale) with Sequential Comparison
    # ========================================================================
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
    
    # Aggiungi dati sequenziali se disponibili
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
    
    # ========================================================================
    # PANEL 6: Occupancy Theory
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 2])
    
    # Calcola occupancy teorica (parametri GPU tipici)
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
    
    # Gestisci il risultato di pie() che può restituire 2 o 3 elementi
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
    else:
        wedges, texts = pie_result
        autotexts = []
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax6.set_title('📊 Theoretical Occupancy', **TITLE_FONT, pad=15)
    
    # ========================================================================
    # Footer con informazioni
    # ========================================================================
    footer_text = (f"Generated from: {len(df)} benchmark runs | "
                  f"Grid sizes: {min(grid_sizes)}–{max(grid_sizes)} | "
                  f"Block sizes: {min(block_sizes)}–{max(block_sizes)} | "
                  f"Best: {best_bs}×{best_bs} threads/block")
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=9, 
            style='italic', color='gray')
    
    return fig

def create_detailed_comparison(df):
    """Grafico dettagliato per ogni grid size"""
    grid_sizes = sorted(df['grid_size'].unique())
    block_sizes = sorted(df['block_size'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Detailed Performance Analysis by Grid Size', 
                 fontsize=20, fontweight='bold', color=COLORS['primary'])
    
    for idx, grid in enumerate(grid_sizes[:4]):  # Primi 4 grid size
        ax = axes[idx // 2, idx % 2]
        data = df[df['grid_size'] == grid]
        
        # Doppio asse y
        ax2 = ax.twinx()
        
        # Plot throughput (barre)
        bars = ax.bar(range(len(block_sizes)), data['throughput_mcells_s'], 
                     alpha=0.7, color=COLORS['primary'], 
                     edgecolor='black', linewidth=1.5, label='Throughput')
        
        # Plot time (linea)
        line = ax2.plot(range(len(block_sizes)), data['time_ms'], 
                       color=COLORS['danger'], marker='o', linewidth=3, 
                       markersize=8, label='Total Time', markeredgecolor='white',
                       markeredgewidth=2)
        
        # Evidenzia il migliore
        best_idx = data['throughput_mcells_s'].idxmax()
        best_bs_idx = list(data.index).index(best_idx)
        bars[best_bs_idx].set_color(COLORS['success'])
        bars[best_bs_idx].set_alpha(1.0)
        bars[best_bs_idx].set_edgecolor('gold')
        bars[best_bs_idx].set_linewidth(3)
        
        ax.set_xlabel('Block Size', **LABEL_FONT)
        ax.set_ylabel('Throughput (M cells/s)', color=COLORS['primary'], **LABEL_FONT)
        ax2.set_ylabel('Total Time (ms)', color=COLORS['danger'], **LABEL_FONT)
        ax.set_title(f'Grid {grid}×{grid}', **TITLE_FONT, pad=10)
        ax.set_xticks(range(len(block_sizes)))
        ax.set_xticklabels([f'{bs}×{bs}' for bs in block_sizes])
        ax.tick_params(axis='y', labelcolor=COLORS['primary'])
        ax2.tick_params(axis='y', labelcolor=COLORS['danger'])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        # Legend combinata
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, 
                 loc='upper right', prop=LEGEND_FONT, framealpha=0.95)
    
    plt.tight_layout()
    return fig

def create_summary_infographic(df, df_seq=None):
    """Infografica riassuntiva stile poster"""
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#f0f0f0')
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.text(0.5, 0.7, 'CUDA BLOCK SIZE OPTIMIZATION', 
                  ha='center', va='center', fontsize=28, 
                  fontweight='bold', color=COLORS['primary'])
    ax_header.text(0.5, 0.3, "Conway's Game of Life Performance Study", 
                  ha='center', va='center', fontsize=16, 
                  style='italic', color='gray')
    
    # Trova statistiche chiave
    best = df.loc[df['throughput_mcells_s'].idxmax()]
    best_bs = int(best['block_size'])
    block_sizes = sorted(df['block_size'].unique())
    
    # Box 1: Winner
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis('off')
    rect = mpatches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=COLORS['success'], 
                                   edgecolor='black', linewidth=3, alpha=0.3)
    ax1.add_patch(rect)
    ax1.text(0.5, 0.75, '🏆', ha='center', va='center', fontsize=40)
    ax1.text(0.5, 0.5, f'{best_bs}×{best_bs}', ha='center', va='center',
            fontsize=36, fontweight='bold', color=COLORS['primary'])
    ax1.text(0.5, 0.25, 'OPTIMAL', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['success'])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Box 2: Peak Throughput
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis('off')
    rect = mpatches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=COLORS['warning'], 
                                   edgecolor='black', linewidth=3, alpha=0.3)
    ax2.add_patch(rect)
    ax2.text(0.5, 0.75, '🚀', ha='center', va='center', fontsize=40)
    ax2.text(0.5, 0.5, f'{best["throughput_mcells_s"]:.1f}', 
            ha='center', va='center', fontsize=32, fontweight='bold')
    ax2.text(0.5, 0.25, 'M cells/sec', ha='center', va='center',
            fontsize=12, style='italic')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Box 3: Speedup vs CPU
    if df_seq is not None:
        # Trova il miglior speedup CUDA vs CPU
        best_grid = int(best['grid_size'])
        seq_match = df_seq[df_seq['grid_size'] == best_grid]
        if not seq_match.empty:
            seq_throughput = seq_match['throughput_mcells_s'].values[0]
            speedup = best['throughput_mcells_s'] / seq_throughput
            speedup_label = f'{speedup:.1f}× vs CPU'
        else:
            baseline_time = df[df['block_size'] == min(block_sizes)]['time_ms'].min()
            best_time = best['time_ms']
            speedup = baseline_time / best_time
            speedup_label = f'{speedup:.1f}× vs BS={min(block_sizes)}'
    else:
        baseline_time = df[df['block_size'] == min(block_sizes)]['time_ms'].min()
        best_time = best['time_ms']
        speedup = baseline_time / best_time
        speedup_label = f'{speedup:.1f}× vs BS={min(block_sizes)}'
    
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    rect = mpatches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=COLORS['secondary'], 
                                   edgecolor='black', linewidth=3, alpha=0.3)
    ax3.add_patch(rect)
    ax3.text(0.5, 0.75, '⚡', ha='center', va='center', fontsize=40)
    ax3.text(0.5, 0.5, speedup_label, ha='center', va='center',
            fontsize=28, fontweight='bold', color='white')
    ax3.text(0.5, 0.25, 'SPEEDUP', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Mini heatmap
    ax4 = fig.add_subplot(gs[2:, :])
    pivot = df.pivot(index='grid_size', columns='block_size', values='throughput_mcells_s')
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', 
               cbar_kws={'label': 'Throughput (M cells/s)'}, 
               ax=ax4, linewidths=2, linecolor='white',
               annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax4.set_xlabel('Block Size (threads per dimension)', **LABEL_FONT)
    ax4.set_ylabel('Grid Size', **LABEL_FONT)
    ax4.set_title('Performance Matrix', **TITLE_FONT, pad=15)
    ax4.set_xticklabels([f'{bs}×{bs}' for bs in pivot.columns])
    ax4.set_yticklabels([f'{gs}×{gs}' for gs in pivot.index], rotation=0)
    
    return fig

def create_cuda_vs_cpu_comparison(df, df_seq):
    """Crea un grafico dedicato al confronto CUDA vs CPU"""
    if df_seq is None:
        return None
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Titolo principale
    fig.suptitle('CUDA vs Sequential CPU: Performance Comparison', 
                 fontsize=22, fontweight='bold', 
                 color=COLORS['primary'], y=0.98)
    
    grid_sizes = sorted(df['grid_size'].unique())
    block_sizes = sorted(df['block_size'].unique())
    best_bs = 16  # Block size ottimale
    worst_bs = 1  # Block size peggiore
    
    # ========================================================================
    # PANEL 1: Throughput Comparison (include 1×1)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(grid_sizes))
    width = 0.22
    
    # Dati CUDA (best block size)
    cuda_best = [df[(df['grid_size'] == g) & (df['block_size'] == best_bs)]['throughput_mcells_s'].values[0]
                       for g in grid_sizes]
    
    # Dati CUDA 1×1
    cuda_1x1 = [df[(df['grid_size'] == g) & (df['block_size'] == worst_bs)]['throughput_mcells_s'].values[0]
                if not df[(df['grid_size'] == g) & (df['block_size'] == worst_bs)].empty else 0
                for g in grid_sizes]
    
    # Dati CPU
    cpu_throughput = [df_seq[df_seq['grid_size'] == g]['throughput_mcells_s'].values[0]
                      if not df_seq[df_seq['grid_size'] == g].empty else 0
                      for g in grid_sizes]
    
    bars1 = ax1.bar(x - width, cuda_best, width, 
                    label=f'CUDA Best (BS={best_bs}×{best_bs})', 
                    color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x, cpu_throughput, width, 
                    label='Sequential CPU', 
                    color=COLORS['warning'], alpha=0.8, edgecolor='black', linewidth=2)
    bars3 = ax1.bar(x + width, cuda_1x1, width, 
                    label=f'CUDA Worst (BS={worst_bs}×{worst_bs})', 
                    color=COLORS['danger'], alpha=0.8, edgecolor='black', linewidth=2, hatch='//')
    
    # Aggiungi valori sulle barre
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar in bars3:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_xlabel('Grid Size', **LABEL_FONT)
    ax1.set_ylabel('Throughput (M cells/s)', **LABEL_FONT)
    ax1.set_title('🚀 Throughput Comparison (Note: 1×1 is SLOWER than CPU!)', **TITLE_FONT, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{g}×{g}' for g in grid_sizes])
    ax1.legend(prop=LEGEND_FONT, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_facecolor('#f8f9fa')
    
    # ========================================================================
    # PANEL 2: Speedup
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    speedups = [cuda_best[i] / cpu_throughput[i] if cpu_throughput[i] > 0 else 0
                for i in range(len(grid_sizes))]
    
    colors = [COLORS['success'] if s > 20 else 
              COLORS['warning'] if s > 10 else 
              COLORS['danger'] for s in speedups]
    
    bars = ax2.barh(range(len(grid_sizes)), speedups, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Aggiungi valori
    for i, (speedup, bar) in enumerate(zip(speedups, bars)):
        ax2.text(speedup + 1, i, f'{speedup:.1f}×', 
                va='center', fontweight='bold', fontsize=12)
    
    ax2.set_yticks(range(len(grid_sizes)))
    ax2.set_yticklabels([f'{g}×{g}' for g in grid_sizes])
    ax2.set_xlabel('Speedup Factor', **LABEL_FONT)
    ax2.set_title('⚡ CUDA Speedup vs CPU', **TITLE_FONT, pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax2.set_facecolor('#f8f9fa')
    
    # ========================================================================
    # PANEL 3: Execution Time Comparison (Log scale)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    cuda_time = [df[(df['grid_size'] == g) & (df['block_size'] == best_bs)]['time_per_gen_ms'].values[0]
                 for g in grid_sizes]
    cpu_time = [df_seq[df_seq['grid_size'] == g]['time_per_gen_ms'].values[0]
                if not df_seq[df_seq['grid_size'] == g].empty else np.nan
                for g in grid_sizes]
    
    ax3.plot(grid_sizes, cuda_time, marker='o', linewidth=3, markersize=10,
            label=f'CUDA (BS={best_bs}×{best_bs})', color=COLORS['success'],
            markeredgecolor='white', markeredgewidth=2)
    ax3.plot(grid_sizes, cpu_time, marker='s', linewidth=3, markersize=10,
            label='Sequential CPU', color=COLORS['danger'],
            markeredgecolor='white', markeredgewidth=2)
    
    ax3.set_xlabel('Grid Size', **LABEL_FONT)
    ax3.set_ylabel('Time per Generation (ms, log scale)', **LABEL_FONT)
    ax3.set_title('⏱️ Execution Time', **TITLE_FONT, pad=15)
    ax3.set_yscale('log')
    ax3.legend(prop=LEGEND_FONT, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--', which='both')
    ax3.set_facecolor('#f8f9fa')
    ax3.set_xticks(grid_sizes)
    ax3.set_xticklabels([f'{g}×{g}' for g in grid_sizes], rotation=45)
    
    # Footer
    avg_speedup = np.mean(speedups)
    max_speedup = np.max(speedups)
    footer_text = (f"Average Speedup: {avg_speedup:.1f}× | "
                  f"Max Speedup: {max_speedup:.1f}× | "
                  f"Best Block Size: {best_bs}×{best_bs}")
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=11, 
            fontweight='bold', color=COLORS['primary'])
    
    return fig

def print_summary_table(df):
    """Stampa tabella riassuntiva ASCII art"""
    print("\n" + "="*90)
    print("║" + " "*88 + "║")
    print("║" + " "*25 + "🏆 PERFORMANCE SUMMARY 🏆" + " "*27 + "║")
    print("║" + " "*88 + "║")
    print("="*90)
    
    grid_sizes = sorted(df['grid_size'].unique())
    block_sizes = sorted(df['block_size'].unique())
    
    # Header
    header = f"{'Grid Size':<15}"
    for bs in block_sizes:
        header += f"│ BS={bs:>2}×{bs:<2} "
    header += "│ Winner"
    print(header)
    print("─"*90)
    
    # Dati
    for grid in grid_sizes:
        row = f"{grid:>4}×{grid:<8}"
        best_throughput = 0
        best_bs = 0
        
        for bs in block_sizes:
            data = df[(df['grid_size'] == grid) & (df['block_size'] == bs)]
            if not data.empty:
                throughput = data['throughput_mcells_s'].values[0]
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_bs = bs
                row += f"│ {throughput:>7.1f} "
            else:
                row += f"│ {'N/A':>7} "
        
        row += f"│ {best_bs}×{best_bs} ⭐"
        print(row)
    
    print("="*90)
    
    # Best overall
    best = df.loc[df['throughput_mcells_s'].idxmax()]
    print(f"\n🎯 OPTIMAL CONFIGURATION:")
    print(f"   Block Size: {int(best['block_size'])}×{int(best['block_size'])} ({int(best['block_size']**2)} threads/block)")
    print(f"   Peak Throughput: {best['throughput_mcells_s']:.2f} M cells/s")
    print(f"   Grid Size: {int(best['grid_size'])}×{int(best['grid_size'])}")
    print("="*90 + "\n")

def main():
    # Path al CSV
    csv_paths = [
        Path('benchmarks/block_size_comparison.csv'),
        Path('../../benchmarks/block_size_comparison.csv'),
        Path('../benchmarks/block_size_comparison.csv')
    ]
    
    seq_csv_paths = [
        Path('benchmarks/benchmark_sequential.csv'),
        Path('../../benchmarks/benchmark_sequential.csv'),
        Path('../benchmarks/benchmark_sequential.csv')
    ]
    
    csv_path = None
    for path in csv_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        print("❌ Error: benchmark CSV not found!")
        print("\nRun the benchmark script first:")
        print("  ./benchmark_block_sizes.sh")
        sys.exit(1)
    
    # Carica dati sequenziali (opzionale)
    seq_csv_path = None
    for path in seq_csv_paths:
        if path.exists():
            seq_csv_path = path
            break
    
    print("\n" + "="*90)
    print("║" + " "*88 + "║")
    print("║" + " "*20 + "🎨 Game of Life - Visual Analysis 🎨" + " "*23 + "║")
    print("║" + " "*88 + "║")
    print("="*90 + "\n")
    
    # Carica dati
    df = load_results(csv_path)
    df_seq = load_sequential_results(seq_csv_path) if seq_csv_path else None
    print()
    
    # Stampa summary
    print_summary_table(df)
    
    # Crea grafici
    print("📊 Generating beautiful plots...\n")
    
    print("  ✓ Creating main dashboard...")
    fig1 = create_main_dashboard(df, df_seq)
    output_path = csv_path.parent / 'block_size_dashboard.png'
    fig1.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    Saved: {output_path}")
    
    print("  ✓ Creating detailed comparison...")
    fig2 = create_detailed_comparison(df)
    output_path = csv_path.parent / 'block_size_detailed.png'
    fig2.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    Saved: {output_path}")
    
    print("  ✓ Creating summary infographic...")
    fig3 = create_summary_infographic(df, df_seq)
    output_path = csv_path.parent / 'block_size_infographic.png'
    fig3.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    
    # Crea grafico CUDA vs CPU se disponibile
    if df_seq is not None:
        print("  ✓ Creating CUDA vs CPU comparison...")
        fig4 = create_cuda_vs_cpu_comparison(df, df_seq)
        if fig4 is not None:
            output_path = csv_path.parent / 'cuda_vs_cpu_comparison.png'
            fig4.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"    Saved: {output_path}")
    
    print("\n✨ All plots generated successfully!")
    print("\n📂 Output files:")
    print(f"   • block_size_dashboard.png    (Main analysis)")
    print(f"   • block_size_detailed.png     (Detailed breakdown)")
    print(f"   • block_size_infographic.png  (Summary poster)")
    if df_seq is not None:
        print(f"   • cuda_vs_cpu_comparison.png  (CUDA vs CPU speedup)")
    
    if df_seq is not None:
        print(f"\n✓ Sequential (CPU) data included for comparison")
    
    print("\n🖥️  Showing interactive plots...")
    plt.show()

if __name__ == "__main__":
    main()