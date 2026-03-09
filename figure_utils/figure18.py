"""
Figure 18: Performance with real-world data.
6-subplot figure: (Approaches, Scenarios, Hardware) x (RSS Loss, AoD Error).

data argument: dict from hardware_results.json
    {"sivers": {scenario: [result_dicts]}, "ibm": {scenario: [result_dicts]}}
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from figure_utils.shared_utils import plot_multi_cdf

REQUIRED_CONFIGS = []


def _build_dataframe(hw_results):
    """Convert hardware_results dict to a combined pandas DataFrame."""
    records = []
    for device_key in ["sivers", "ibm"]:
        device_data = hw_results.get(device_key, {})
        for scenario, result_list in device_data.items():
            for r in result_list:
                try:
                    records.append({
                        "our_rss_loss": float(r["our_rss_loss"]),
                        "baseline_rss_loss": float(r["baseline_rss_loss"]),
                        "our_angle_error": float(r["our_angle_error"]),
                        "baseline_angle_error": float(r["baseline_angle_error"]),
                        "condition": scenario,
                        "device": device_key,
                    })
                except (KeyError, ValueError, TypeError):
                    continue
    return pd.DataFrame(records)


def plot(hw_results, save_path):
    """
    Plot Figure 18: Performance with real-world data.

    Args:
        hw_results: dict from hardware_results.json
        save_path: path to save the figure
    """
    combined = _build_dataframe(hw_results)

    if combined.empty:
        print("Warning: No valid hardware evaluation results found for Figure 18.")
        return

    figure_1_dict = {
        "BeamFormer": combined["our_rss_loss"].values,
        "Fine Sweep": combined["baseline_rss_loss"].values,
    }
    figure_2_dict = {
        "BeamFormer": combined["our_angle_error"].values,
        "Fine Sweep": combined["baseline_angle_error"].values,
    }

    figure_3_dict = {
        "Indoor-LoS": combined[combined["condition"] == "indoor-los"]["our_rss_loss"].values,
        "Indoor-NLoS": combined[combined["condition"] == "indoor-nlos"]["our_rss_loss"].values,
        "Outdoor-LoS": combined[combined["condition"] == "outdoor-los"]["our_rss_loss"].values,
    }
    figure_3_dict = {k: v for k, v in figure_3_dict.items() if len(v) > 0}

    figure_4_dict = {
        "Indoor-LoS": combined[combined["condition"] == "indoor-los"]["our_angle_error"].values,
        "Indoor-NLoS": combined[combined["condition"] == "indoor-nlos"]["our_angle_error"].values,
        "Outdoor-LoS": combined[combined["condition"] == "outdoor-los"]["our_angle_error"].values,
    }
    figure_4_dict = {k: v for k, v in figure_4_dict.items() if len(v) > 0}

    figure_5_dict = {
        k: combined[combined["device"] == k.lower()]["our_rss_loss"].values
        for k in ["Sivers", "IBM"]
        if len(combined[combined["device"] == k.lower()]) > 0
    }
    figure_6_dict = {
        k: combined[combined["device"] == k.lower()]["our_angle_error"].values
        for k in ["Sivers", "IBM"]
        if len(combined[combined["device"] == k.lower()]) > 0
    }

    data_to_plot = [
        figure_1_dict, figure_2_dict,
        figure_3_dict, figure_4_dict,
        figure_5_dict, figure_6_dict,
    ]
    subtitle_name = [
        "(a) Approaches", "(b) Approaches",
        "(c) Scenarios", "(d) Scenarios",
        "(e) Hardware", "(f) Hardware",
    ]
    xlabel = [
        "RSS Loss (dB)", r"Dominant AoD Error ($^\circ$)",
        "RSS Loss (dB)", r"Dominant AoD Error ($^\circ$)",
        "RSS Loss (dB)", r"Dominant AoD Error ($^\circ$)",
    ]

    plot_multi_cdf(
        data_to_plot,
        subplot_shape=(3, 2),
        main_title=None,
        subplot_titles=subtitle_name,
        xlabel=xlabel,
        ylabel="CDF",
        x_limits=(0, 10),
        show_grid=True,
        save_path=save_path,
    )
