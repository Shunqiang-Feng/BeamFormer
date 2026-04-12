"""
Figure 15: Multi Path Prediction Accuracy.
CDF of RSS prediction error for the 1st, 2nd, and 3rd strongest paths.
Evaluation uses the ct_xa16_large_ft model on homeoffice 28G 16x16 indoor data.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf

REQUIRED_CONFIGS = []  # special case - uses dedicated multi_path_eval_new runner

_DEFAULT_RESULTS_PATH = os.path.join(
    _PROJECT_ROOT, "eval_results", "multi_path_eval_new", "results.json"
)


def load_from_json(results_path):
    """
    Load multi-path evaluation results from JSON.

    Args:
        results_path: path to JSON produced by beamformer.multi_path_eval_new

    Returns:
        pd.DataFrame with columns path1_error_db, path2_error_db, path3_error_db
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Multi-path results JSON not found: {results_path}\n"
            "Run with --data_source from_scratch to generate evaluation results."
        )
    df = pd.read_json(results_path)
    print(f"[Figure 15] Loaded {len(df)} samples from {results_path}")
    return df


def run(num_samples=None, batch_size=1, save_path=None):
    """
    Run multi-path RSS error evaluation using beamformer.multi_path_eval_new.

    Args:
        num_samples: max number of test samples (None = all)
        batch_size:  inference batch size
        save_path:   where to save the JSON results
                     (default: eval_results/multi_path_eval_new/results.json)

    Returns:
        str: path to the saved JSON file
    """
    from beamformer.multi_path_eval_new import run_multi_path_eval_new

    if save_path is None:
        save_path = _DEFAULT_RESULTS_PATH

    df = run_multi_path_eval_new(num_samples=num_samples, batch_size=batch_size)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_json(save_path, orient="records", indent=2)
    print(f"[Figure 15] Results saved to {save_path}")

    for k in range(1, 4):
        col = f"path{k}_error_db"
        if col in df.columns:
            vals = df[col].dropna().abs()
            print(
                f"  Path {k}: n={len(vals)}, "
                f"mean_abs={vals.mean():.3f} dB, "
                f"median={vals.median():.3f} dB, "
                f"90th={vals.quantile(0.9):.3f} dB"
            )

    return save_path


def plot(data, save_path, x_max=20):
    """
    Plot Figure 15: Multi-path prediction accuracy CDF.

    Args:
        data: dict with key 'results_path' (str path to JSON),
              OR a pd.DataFrame with path1_error_db ... columns,
              OR a str path directly.
        save_path: output figure path
        x_max: x-axis upper limit in dB
    """
    if isinstance(data, str):
        df = load_from_json(data)
    elif isinstance(data, dict):
        df = load_from_json(data["results_path"])
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    data_dict = {}
    for k, label in enumerate(["1st Peak", "2nd Peak", "3rd Peak"], start=1):
        col = f"path{k}_error_db"
        if col not in df.columns:
            print(f"  Warning: column {col} not found, skipping.")
            continue
        vals = np.abs(df[col].dropna().values.astype(float))
        data_dict[label] = vals
        print(
            f"  {label}: n={len(vals)}, "
            f"mean_abs={vals.mean():.3f} dB, "
            f"median={np.median(vals):.3f} dB, "
            f"90th={np.percentile(vals, 90):.3f} dB"
        )

    if not data_dict:
        raise ValueError("No valid path error columns found in results.")

    plot_multi_cdf(
        data_list=[data_dict],
        subplot_shape=(1, 1),
        save_path=save_path,
        xlabel="RSS Error",
        ylabel="CDF",
        x_units="dB",
        x_limits=(0, x_max),
        y_limits=(0.0, 1.0),
        legend_position="lower right",
        basic_fontsize=16,
        h_w_ratio=0.85,
    )
