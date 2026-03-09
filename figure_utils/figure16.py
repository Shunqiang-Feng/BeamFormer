"""
Figure 16: Comparison of reference beam settings.
Plots 1x1 CDF for different beam weight configurations (optimized vs directional).
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf, calculate_angle_error

REQUIRED_CONFIGS = ["compare_beam_settings"]

NAME_SWITCH_TABLE = {
    'Directional Beam Weight [Full Array]': 'Narrow Pencil Beams',
    'Directional Beam Weight [4x4 Subarray]': 'Wide Pencil Beams',
    'ML-Optimized Weight': 'Optimized Beams',
    'Random Weight': 'Random Weight'
}

_KEYS_TO_PLOT = [
    'ML-Optimized Weight',
    'Directional Beam Weight [Full Array]',
    'Directional Beam Weight [4x4 Subarray]',
]


def _fun_name_switch(name):
    return NAME_SWITCH_TABLE.get(name, name)


def _get_rss_error(df):
    return abs(df['rss_at_gt'].values - df['max_rss'].values)


def _add_angle_error_all(data_dict):
    """Add angle_error column to all entries that have pred_phi/pred_theta."""
    result = {}
    for key_name, df in data_dict.items():
        df = df.copy()
        if 'pred_phi' in df.columns and 'pred_theta' in df.columns:
            df['angle_error'] = calculate_angle_error(
                df['gt_phi'], df['gt_theta'],
                df['pred_phi'], df['pred_theta']
            )
        result[key_name] = df
    return result


def plot(data, save_path):
    """
    Plot Figure 16: Beam settings CDF.

    Args:
        data: dict mapping config_name -> dict of {model_name -> DataFrame}
              Key: 'compare_beam_settings'
        save_path: path to save the resulting figure
    """
    compare_beam_json = data["compare_beam_settings"]
    compare_beam_json = _add_angle_error_all(compare_beam_json)

    compare_beam_data = {}
    for key in _KEYS_TO_PLOT:
        if key in compare_beam_json:
            compare_beam_data[_fun_name_switch(key)] = _get_rss_error(compare_beam_json[key])

    data_list = [compare_beam_data]

    plot_multi_cdf(
        data_list,
        subplot_shape=(1, 1),
        main_title=None,
        subplot_titles=None,
        x_limits=(0, 10),
        xlabel=["RSS Error (dB)"],
        ylabel="CDF",
        show_grid=True,
        save_path=save_path
    )
