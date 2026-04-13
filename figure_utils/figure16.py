"""
Figure 16: Comparison of model latencies.
Reads from pre-computed benchmark CSV files and plots latency vs query number.
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

REQUIRED_CONFIGS = []  # special case - uses CSV files

_CSV_DIR = os.path.join(_PROJECT_ROOT, "eval_results", "running_time_comparison", "cached")
_PI_CSV = "model_benchmark_results_pi.csv"
_TRANSFORMER_CSV = "model_benchmark_results_transformer.csv"

# Styling constants (match multi_cdf style)
BASIC_FONTSIZE = 16
H_W_RATIO = 0.85
PLOT_WIDTH = 3.33
PLOT_HEIGHT = 3.33 * H_W_RATIO
LEFT_MARGIN = 0.8
RIGHT_MARGIN = 0.3
BOTTOM_MARGIN = 0.7
TOP_MARGIN = 0.5
FIG_WIDTH = LEFT_MARGIN + PLOT_WIDTH + RIGHT_MARGIN
FIG_HEIGHT = BOTTOM_MARGIN + PLOT_HEIGHT + TOP_MARGIN
FONT_SIZE_LABEL = BASIC_FONTSIZE
FONT_SIZE_TICK = BASIC_FONTSIZE - 2
FONT_SIZE_LEGEND = BASIC_FONTSIZE - 2
LINE_WIDTH = 2.0
MARKER_SIZE = 4
MARKER_EVERY = 1
DPI = 300
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']


def load_from_csv(csv_dir=None):
    """
    Load pre-computed benchmark CSVs.

    Args:
        csv_dir: directory containing the CSV files. Defaults to running_time_comparison/.

    Returns:
        dict with keys 'pi' and 'transformer', each a pandas DataFrame
    """
    if csv_dir is None:
        csv_dir = _CSV_DIR

    pi_path = os.path.join(csv_dir, _PI_CSV)
    transformer_path = os.path.join(csv_dir, _TRANSFORMER_CSV)

    if not os.path.exists(pi_path):
        raise FileNotFoundError(
            f"BeamFormer benchmark CSV not found: {pi_path}\n"
            f"Please ensure the CSV files are in {csv_dir}/"
        )
    if not os.path.exists(transformer_path):
        raise FileNotFoundError(
            f"Transformer benchmark CSV not found: {transformer_path}\n"
            f"Please ensure the CSV files are in {csv_dir}/"
        )

    df_pi = pd.read_csv(pi_path)
    df_transformer = pd.read_csv(transformer_path)

    print(f"Loaded benchmark CSVs from {csv_dir}")
    return {"pi": df_pi, "transformer": df_transformer}


def plot(data, save_path):
    """
    Plot Figure 15: Model latency comparison (BeamFormer vs Transformer).

    Args:
        data: dict with keys 'pi' (DataFrame) and 'transformer' (DataFrame)
        save_path: path to save the resulting figure
    """
    df_pi = data["pi"]
    df_transformer = data["transformer"]

    df_pi_64 = df_pi[df_pi['sample_number'] == 64].sort_values('query_number')
    df_transformer_64 = df_transformer[df_transformer['sample_number'] == 64].sort_values('query_number')

    query_numbers = df_pi_64['query_number'].values
    time_pi = df_pi_64['time_mean_ms'].values
    time_transformer = df_transformer_64['time_mean_ms'].values
    time_ratio = time_transformer / time_pi

    print(f"Data loaded successfully!")
    print(f"Query numbers: {query_numbers}")

    # Configure matplotlib
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['font.size'] = BASIC_FONTSIZE
    mpl.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
    mpl.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
    mpl.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
    mpl.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    mpl.rcParams['figure.dpi'] = DPI
    mpl.rcParams['savefig.dpi'] = DPI
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['lines.linewidth'] = LINE_WIDTH
    mpl.rcParams['axes.spines.top'] = True
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.spines.bottom'] = True
    mpl.rcParams['axes.spines.left'] = True

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    left = LEFT_MARGIN / FIG_WIDTH
    bottom = BOTTOM_MARGIN / FIG_HEIGHT
    width = PLOT_WIDTH / FIG_WIDTH
    height = PLOT_HEIGHT / FIG_HEIGHT

    ax = fig.add_axes([left, bottom, width, height])

    x_positions = np.arange(len(query_numbers))

    # Plot Transformer line
    ax.plot(x_positions, time_transformer,
            marker=MARKERS[0],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            color=COLORS[0],
            linestyle=LINE_STYLES[0],
            markerfacecolor=COLORS[0],
            markevery=MARKER_EVERY,
            label='Transformer',
            alpha=0.8,
            zorder=3)

    # Plot BeamFormer line
    ax.plot(x_positions, time_pi,
            marker=MARKERS[1],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            color=COLORS[1],
            linestyle=LINE_STYLES[1],
            markerfacecolor=COLORS[1],
            markevery=MARKER_EVERY,
            label='BeamFormer',
            alpha=0.8,
            zorder=3)

    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

    # X-axis formatting (unit: K)
    ax.set_xticks(x_positions)
    x_labels = []
    for qn in query_numbers:
        k_value = qn / 1000
        if k_value == int(k_value):
            x_labels.append(f'{int(k_value)}')
        else:
            x_labels.append(f'{k_value:.1f}'.rstrip('0').rstrip('.'))

    ax.set_xticklabels(x_labels, fontsize=FONT_SIZE_TICK, family='Arial')

    # Y-axis - log scale
    ax.set_yscale('log')
    ax.set_ylim([1, 1500])
    y_ticks = [1, 10, 100, 1000]
    y_labels = ['1', '10', r'$10^2$', r'$10^3$']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=FONT_SIZE_TICK)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    ax.set_xlabel('Query Number (K)', fontsize=FONT_SIZE_LABEL, family='Arial')
    ax.set_ylabel('Latency (ms)', fontsize=FONT_SIZE_LABEL, family='Arial')

    legend = ax.legend(loc='upper left',
                       fontsize=FONT_SIZE_LEGEND,
                       frameon=True,
                       framealpha=0.95,
                       fancybox=False,
                       shadow=False)
    for text in legend.get_texts():
        text.set_fontfamily('Arial')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')

    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=DPI)
    print(f"Figure saved to {save_path}")

    # Print statistics
    print(f"\n{'='*60}")
    print("Performance Analysis:")
    print(f"  Average ratio: {time_ratio.mean():.2f}x")
    print(f"  BeamFormer is {time_ratio.mean():.2f}x faster on average")
    print("\nDetailed Breakdown:")
    print("-" * 60)
    print(f"{'Query':<10} {'Transformer':<15} {'BeamFormer':<15} {'Ratio':<10}")
    print("-" * 60)
    for qn, t_trans, t_pi, ratio in zip(query_numbers, time_transformer, time_pi, time_ratio):
        print(f"{int(qn):<10} {t_trans:<15.3f} {t_pi:<15.3f} {ratio:<10.2f}x")
    print("="*60)

    plt.close(fig)
