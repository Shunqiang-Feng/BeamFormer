"""
Figure 13: Impact of model parameters (ablation study).
Plots 2x2 CDF for # Reference Beams, # Feature Dimension, # Processing Layers, # Latent Tokens.
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf

REQUIRED_CONFIGS = ["ablation_study"]


def _get_rss_error(df):
    return abs(df['rss_at_gt'].values - df['max_rss'].values)


def plot(data, save_path):
    """
    Plot Figure 13: Ablation study CDFs.

    Args:
        data: dict mapping config_name -> dict of {model_name -> DataFrame}
              Key: 'ablation_study'
        save_path: path to save the resulting figure
    """
    ablation_df = data["ablation_study"]

    ablation_data = [
        # sample_num
        {
            "64": _get_rss_error(ablation_df['Baseline']),
            "32": _get_rss_error(ablation_df['Sample-32']),
            "16": _get_rss_error(ablation_df['Sample-16']),
            "8": _get_rss_error(ablation_df['Sample-8']),
        },
        # dim
        {
            "1024": _get_rss_error(ablation_df['Baseline']),
            "512": _get_rss_error(ablation_df["Dim-512"]),
            "256": _get_rss_error(ablation_df["Dim-256"]),
        },
        # Depth
        {
            "8": _get_rss_error(ablation_df['Baseline']),
            "4": _get_rss_error(ablation_df["Depth-4"]),
            "2": _get_rss_error(ablation_df["Depth-2"]),
            "1": _get_rss_error(ablation_df["Depth-1"]),
        },
        # latent-num
        {
            "64": _get_rss_error(ablation_df['Baseline']),
            "32": _get_rss_error(ablation_df["Latent-32"]),
            "16": _get_rss_error(ablation_df["Latent-16"]),
        },
    ]

    subplot_titles = [
        "(a) # Reference Beams",
        "(b) # Feature Dimension",
        "(c) # Processing Layers",
        "(d) # Latent Tokens"
    ]

    # Print summary statistics
    ablation_names = ["Sample Num", "Dim", "Depth", "Latent Num"]
    for i, (name, data_dict) in enumerate(zip(ablation_names, ablation_data)):
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        print(f"{'Key':<10} {'50% (Median)':<15} {'90%':<15}")
        print(f"{'-'*40}")
        for key, values in data_dict.items():
            p50 = np.percentile(values, 50)
            p90 = np.percentile(values, 90)
            print(f"{key:<10} {p50:<15.4f} {p90:<15.4f}")

    plot_multi_cdf(
        data_list=ablation_data,
        subplot_shape=(2, 2),
        main_title=None,
        ylabel="CDF",
        subplot_titles=subplot_titles,
        xlabel="RSS Error (dB)",
        show_grid=True,
        x_limits=(0, 10),
        save_path=save_path
    )
