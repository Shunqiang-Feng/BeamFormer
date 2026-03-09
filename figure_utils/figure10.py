"""
Figure 10: Overall performance of different approaches.
Plots 1x2 CDF with RSS Loss and AoD Error, combining indoor and outdoor scenarios.
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf, calculate_angle_error

REQUIRED_CONFIGS = ["compare_vs_baselines_indoor", "compare_vs_baselines_outdoor"]

NAME_SWITCH_TABLE = {
    'Our Method': 'BeamFormer',
    '802ad': 'Fine Sweep',
    'SectorSweep': 'Coarse Sweep',
    'Hierarchical': 'Hier. Sweep',
    '2ACE': '2ACE',
    'AgileLink': 'AgileLink'
}


def _fun_name_switch(name):
    return NAME_SWITCH_TABLE.get(name, name)


def _combine_dict_arrays(dict1, dict2):
    """
    Combine two dictionaries that have the same keys and numpy array values.

    Args:
        dict1: Dictionary with numpy array values
        dict2: Dictionary with numpy array values (same keys as dict1)

    Returns:
        Dictionary with combined numpy arrays for each key
    """
    combined_dict = {}
    for key in dict1.keys():
        if key in dict2:
            combined_dict[key] = np.concatenate([dict1[key], dict2[key]])
        else:
            raise ValueError("Error Code 2025-08-25-001")
    for key in dict2.keys():
        if key not in combined_dict:
            raise ValueError("Error Code 2025-08-25-002")
    return combined_dict


def _add_angle_error(data_dict):
    """Add angle_error column to 'Our Method' DataFrame."""
    key_name = 'Our Method'
    if key_name in data_dict:
        df = data_dict[key_name]
        df = df.copy()
        df['angle_error'] = calculate_angle_error(
            df['gt_phi'], df['gt_theta'],
            df['pred_phi'], df['pred_theta']
        )
        data_dict = dict(data_dict)
        data_dict[key_name] = df
    return data_dict


def plot(data, save_path):
    """
    Plot Figure 10: RSS Loss and AoD Error CDFs (indoor + outdoor combined).

    Args:
        data: dict mapping config_name -> dict of {model_name -> DataFrame}
              Keys should be 'compare_vs_baselines_indoor' and 'compare_vs_baselines_outdoor'
        save_path: path to save the resulting figure
    """
    indoor_pd = data["compare_vs_baselines_indoor"]
    outdoor_pd = data["compare_vs_baselines_outdoor"]

    # Add angle error for 'Our Method'
    indoor_pd = _add_angle_error(indoor_pd)
    outdoor_pd = _add_angle_error(outdoor_pd)

    # RSS Loss dicts (all methods)
    indoor_loss = {}
    for key in indoor_pd.keys():
        confirmed_key = _fun_name_switch(key)
        indoor_loss[confirmed_key] = indoor_pd[key]['loss'].values

    outdoor_loss = {}
    for key in outdoor_pd.keys():
        confirmed_key = _fun_name_switch(key)
        outdoor_loss[confirmed_key] = outdoor_pd[key]['loss'].values

    # Angle Error dicts (all methods that have it)
    indoor_angle_error = {}
    for key in indoor_pd.keys():
        confirmed_key = _fun_name_switch(key)
        if 'angle_error' in indoor_pd[key].columns:
            indoor_angle_error[confirmed_key] = indoor_pd[key]['angle_error'].values
        else:
            indoor_angle_error[confirmed_key] = np.full(len(indoor_pd[key]), np.nan)

    outdoor_angle_error = {}
    for key in outdoor_pd.keys():
        confirmed_key = _fun_name_switch(key)
        if 'angle_error' in outdoor_pd[key].columns:
            outdoor_angle_error[confirmed_key] = outdoor_pd[key]['angle_error'].values
        else:
            outdoor_angle_error[confirmed_key] = np.full(len(outdoor_pd[key]), np.nan)

    # Combine indoor and outdoor
    combined_loss = _combine_dict_arrays(indoor_loss, outdoor_loss)
    combined_angle_error = _combine_dict_arrays(indoor_angle_error, outdoor_angle_error)

    # Remove methods with all-NaN angle errors
    combined_angle_error_clean = {
        k: v for k, v in combined_angle_error.items() if not np.all(np.isnan(v))
    }

    subplot_titles = ["(a) RSS Loss", "(b) Dominant AoD Error"]
    vs_baseline_data = [combined_loss, combined_angle_error_clean]
    xlabel = [
        "RSS Loss (dB)",
        r"AoD Error ($^\circ$)",
    ]

    plot_multi_cdf(
        vs_baseline_data,
        subplot_shape=(1, 2),
        main_title=None,
        subplot_titles=subplot_titles,
        xlabel=xlabel,
        ylabel="CDF",
        show_grid=True,
        x_limits=[(0, 20), (0, 20)],
        save_path=save_path
    )

    # Print summary statistics for BeamFormer
    beamformer_key = 'BeamFormer'
    if beamformer_key in combined_loss:
        rss_loss_90th = np.percentile(combined_loss[beamformer_key], 90)
        print(f"BeamFormer @ 90% CDF:")
        print(f"  RSS Loss: {rss_loss_90th:.2f} dB")
    if beamformer_key in combined_angle_error_clean:
        aod_error_90th = np.percentile(combined_angle_error_clean[beamformer_key], 90)
        print(f"  AoD Error: {aod_error_90th:.2f} degrees")
