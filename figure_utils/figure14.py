"""
Figure 14: Comparison of positional encoders.
CDF of RSS Error comparing Array Factor, 2D Positional, and Concat position encodings.
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf

REQUIRED_CONFIGS = ["cross_encoder"]

# Model name -> display label (ordered: Beam Pattern, Beam Weight, Beam Direction)
NAME_SWITCH = {
    "Array Factor": "Beam Pattern",
    "Concat": "Beam Weight",
    "2D Positional": "Beam Direction",
}


def _get_rss_error(df):
    return np.abs(df["rss_at_gt"].values - df["max_rss"].values)


def plot(data, save_path):
    """
    Plot Figure 14: Comparison of positional encoders.

    Args:
        data: dict mapping config_name -> {model_name -> DataFrame}
              Key: 'cross_encoder'
        save_path: path to save the resulting figure
    """
    df_dict = data["cross_encoder"]

    cdf_data = {
        label: _get_rss_error(df_dict[model_name])
        for model_name, label in NAME_SWITCH.items()
        if model_name in df_dict
    }

    plot_multi_cdf(
        [cdf_data],
        subplot_shape=(1, 1),
        main_title=None,
        xlabel="RSS Error (dB)",
        ylabel="CDF",
        x_limits=[(0, 10)],
        show_grid=True,
        save_path=save_path,
    )
