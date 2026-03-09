"""
Figure 12: Impact of scene configurations.
Plots 2x2 CDF for cross-frequency, cross-scenario, cross-antenna-pattern, and cross-SNR.
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf

REQUIRED_CONFIGS = ["cross_frequency", "cross_scenarios", "cross_pattern", "cross_snr"]


def _fun_name_switch_freq(name):
    NAME_SWITCH_TABLE = {
        'indoor-model-14G': '14 GHz',
        'indoor-model-28G': '28 GHz',
        'indoor-model-60G': '60 GHz',
    }
    return NAME_SWITCH_TABLE.get(name, name)


def _fun_name_switch_scenarios(name):
    name = name.removeprefix("indoor-model-")
    NAME_SWITCH_TABLE = {
        'homeoffice': 'Home & Office',
        'classroom': 'Classroom',
        'uva-comm': 'Campus',
        'uva-radar': 'Campus Sensing',
        'rural-comm': 'Rural',
        'rural-radar': 'Rural Sensing',
        'nyc-comm': 'Urban',
        'nyc-radar': 'Urban Sensing',
    }
    return NAME_SWITCH_TABLE.get(name, name)


def plot(data, save_path):
    """
    Plot Figure 12: Cross-configuration CDFs.

    Args:
        data: dict mapping config_name -> dict of {model_name -> DataFrame}
              Keys: 'cross_frequency', 'cross_scenarios', 'cross_pattern', 'cross_snr'
        save_path: path to save the resulting figure
    """
    # Cross-frequency
    cross_frequency_pd = data["cross_frequency"]
    cross_frequency_data = {}
    for i_key in cross_frequency_pd.keys():
        cross_frequency_data[_fun_name_switch_freq(i_key)] = cross_frequency_pd[i_key]['loss'].values

    # Cross-scenarios (exclude radar and deepmimo)
    cross_scenarios_pd = data["cross_scenarios"]
    cross_scenarios_data = {}
    for i_key in cross_scenarios_pd.keys():
        if ('radar' not in i_key) and ('deepmimo' not in i_key):
            cross_scenarios_data[_fun_name_switch_scenarios(i_key)] = cross_scenarios_pd[i_key]['loss'].values

    # Cross-pattern
    cross_pattern_pd = data["cross_pattern"]
    cross_pattern_data = {}
    for i_key in cross_pattern_pd.keys():
        cross_pattern_data[i_key] = cross_pattern_pd[i_key]['loss'].values

    # Cross-SNR
    cross_snr_pd = data["cross_snr"]
    cross_snr_data = {}
    for i_key in cross_snr_pd.keys():
        cross_snr_data[i_key.replace("-n", ": -") + " dB"] = cross_snr_pd[i_key]['loss'].values

    subplot_titles = ["(a) Cross-Frequency", "(b) Cross-Scenario", "(c) Cross-Antenna", "(d) Cross-SNR"]
    scale_and_robust_data = [cross_frequency_data, cross_scenarios_data, cross_pattern_data, cross_snr_data]

    plot_multi_cdf(
        scale_and_robust_data,
        subplot_shape=(2, 2),
        main_title=None,
        subplot_titles=subplot_titles,
        xlabel="RSS Loss (dB)",
        ylabel="CDF",
        x_limits=(0, 10),
        show_grid=True,
        save_path=save_path
    )
