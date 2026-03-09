"""
Figure 17: Comparison of power estimators.
CDF of RSS Error comparing ARN power estimator vs Normal Distribution baseline,
for indoor and outdoor scenarios.
"""
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf

REQUIRED_CONFIGS = ["test_arn_performance_outdoor"]

# Peak scale statistics per scenario (used by reproduce_figures.py when running from scratch)
PEAK_SCALE_PARAMS = {
    "test_arn_performance_outdoor": {"peak_mean": 31.8946, "peak_std": 12.2739},
}

KEY_NAME_SWITCH = {
    "rss_at_gt":              "Power Estimator",
    "rss_at_gt_random_peak":  "Normal Distribution",
}


def _build_cdf_data(df_dict):
    """
    Build the CDF data dict for one scenario.

    Args:
        df_dict: {model_name -> DataFrame}  (only "Our Method" is used)

    Returns:
        dict {"Power Estimator": np.array, "Normal Distribution": np.array}
    """
    our_method = df_dict["Our Method"]
    result = {}
    for key, label in KEY_NAME_SWITCH.items():
        result[label] = np.abs(our_method[key].values - our_method["max_rss"].values)
    return result


def plot(data, save_path):
    """
    Plot Figure 17: Power estimator comparison (outdoor CDF).

    Args:
        data: dict mapping config_name -> {model_name -> DataFrame}
              Key: 'test_arn_performance_outdoor'
        save_path: path to save the resulting figure
    """
    outdoor_data = _build_cdf_data(data["test_arn_performance_outdoor"])

    plot_multi_cdf(
        [outdoor_data],
        subplot_shape=(1, 1),
        main_title=None,
        xlabel="RSS Error (dB)",
        ylabel="CDF",
        x_limits=[(0, 10)],
        show_grid=True,
        save_path=save_path,
    )
