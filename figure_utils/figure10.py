"""
Figure 10: Overall performance of different approaches.
Plots 2x2 CDF: (RSS Loss, AoD Error) x (LoS, NLoS), combining indoor and outdoor.
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf, calculate_angle_error

REQUIRED_CONFIGS = [
    "compare_vs_baselines_indoor",
    "compare_vs_baselines_outdoor",
    "compare_vs_baselines_indoor_nlos",
    "compare_vs_baselines_outdoor_nlos",
]

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
    """Combine two dicts with the same keys by concatenating numpy arrays."""
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
    """Add angle_error column to any DataFrame that has pred_phi/pred_theta but no angle_error."""
    data_dict = dict(data_dict)
    for key, df in data_dict.items():
        if 'angle_error' not in df.columns and {'pred_phi', 'pred_theta', 'gt_phi', 'gt_theta'}.issubset(df.columns):
            df = df.copy()
            df['angle_error'] = calculate_angle_error(
                df['gt_phi'], df['gt_theta'],
                df['pred_phi'], df['pred_theta']
            )
            data_dict[key] = df
    return data_dict


def _build_loss_and_angle_dicts(pd_dict):
    """
    Build (loss_dict, angle_error_dict) from a model_name -> DataFrame mapping.
    Methods without angle prediction get NaN-filled arrays.
    """
    loss_dict  = {}
    angle_dict = {}
    for key, df in pd_dict.items():
        label = _fun_name_switch(key)
        loss_dict[label] = df['loss'].values
        if 'angle_error' in df.columns:
            angle_dict[label] = df['angle_error'].values
        else:
            angle_dict[label] = np.full(len(df), np.nan)
    return loss_dict, angle_dict


def plot(data, save_path):
    """
    Plot Figure 10: RSS Loss and AoD Error CDFs (LoS and NLoS, indoor + outdoor).

    Args:
        data: dict mapping config_name -> dict of {model_name -> DataFrame}
              Keys: 'compare_vs_baselines_indoor', 'compare_vs_baselines_outdoor',
                    'compare_vs_baselines_indoor_nlos', 'compare_vs_baselines_outdoor_nlos'
        save_path: path to save the resulting figure
    """
    # LoS
    indoor_pd      = _add_angle_error(data["compare_vs_baselines_indoor"])
    outdoor_pd     = _add_angle_error(data["compare_vs_baselines_outdoor"])
    # NLoS
    indoor_nlos_pd  = _add_angle_error(data["compare_vs_baselines_indoor_nlos"])
    outdoor_nlos_pd = _add_angle_error(data["compare_vs_baselines_outdoor_nlos"])

    indoor_loss,      indoor_angle      = _build_loss_and_angle_dicts(indoor_pd)
    outdoor_loss,     outdoor_angle     = _build_loss_and_angle_dicts(outdoor_pd)
    indoor_nlos_loss, indoor_nlos_angle = _build_loss_and_angle_dicts(indoor_nlos_pd)
    outdoor_nlos_loss,outdoor_nlos_angle= _build_loss_and_angle_dicts(outdoor_nlos_pd)

    los_loss        = _combine_dict_arrays(indoor_loss,      outdoor_loss)
    los_angle       = _combine_dict_arrays(indoor_angle,     outdoor_angle)
    nlos_loss       = _combine_dict_arrays(indoor_nlos_loss, outdoor_nlos_loss)
    nlos_angle      = _combine_dict_arrays(indoor_nlos_angle,outdoor_nlos_angle)

    # Drop methods with all-NaN angle errors
    los_angle_clean  = {k: v for k, v in los_angle.items()  if not np.all(np.isnan(v))}
    nlos_angle_clean = {k: v for k, v in nlos_angle.items() if not np.all(np.isnan(v))}

    subplot_titles = [
        "(a) RSS Loss (LoS)",
        "(b) Dominant AoD Error (LoS)",
        "(c) RSS Loss (NLoS)",
        "(d) Dominant AoD Error (NLoS)",
    ]
    vs_baseline_data = [los_loss, los_angle_clean, nlos_loss, nlos_angle_clean]
    xlabel = [
        "RSS Loss (dB)",
        r"AoD Error ($^\circ$)",
        "RSS Loss (dB)",
        r"AoD Error ($^\circ$)",
    ]

    plot_multi_cdf(
        vs_baseline_data,
        subplot_shape=(2, 2),
        main_title=None,
        subplot_titles=subplot_titles,
        xlabel=xlabel,
        ylabel="CDF",
        show_grid=True,
        x_limits=[(0, 20), (0, 20), (0, 20), (0, 20)],
        save_path=save_path,
        only_1_legend=True,
    )

    # Print summary statistics for BeamFormer
    bf = 'BeamFormer'
    for tag, loss_d, angle_d in [("LoS", los_loss, los_angle_clean),
                                   ("NLoS", nlos_loss, nlos_angle_clean)]:
        if bf in loss_d:
            print(f"BeamFormer {tag} @ 90% CDF:")
            print(f"  RSS Loss:  {np.percentile(loss_d[bf], 90):.2f} dB")
        if bf in angle_d:
            print(f"  AoD Error: {np.percentile(angle_d[bf], 90):.2f} degrees")
