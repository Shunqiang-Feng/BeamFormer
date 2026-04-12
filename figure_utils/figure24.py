"""
Figure 24: Example spectrum with IBM SDR.
Uses compare_scan_with_pred on
csi-dataset/realworld_ibm/indoor-los/beamtable_20251203_193653/experiment_special.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REQUIRED_CONFIGS = []

# Default data path for Figure 24
DEFAULT_EXPERIMENT_PATH = os.path.join(
    _PROJECT_ROOT,
    "csi-dataset/realworld_ibm/indoor-los/beamtable_20251203_193653/experiment_special"
)


def plot(save_path, experiment_path=None):
    """
    Plot Figure 24: IBM SDR spectrum example.

    Args:
        save_path: path to save the figure
        experiment_path: path to IBM experiment_special folder
    """
    from beamformer.hardware_inference import plot_ibm_spectrum

    if experiment_path is None:
        experiment_path = DEFAULT_EXPERIMENT_PATH

    if not os.path.exists(experiment_path):
        raise FileNotFoundError(
            f"IBM data not found: {experiment_path}\n"
            f"Expected at: csi-dataset/realworld_ibm/indoor-los/beamtable_20251203_193653/experiment_special"
        )

    print(f"[Figure 24] Generating IBM spectrum from: {experiment_path}")
    fig = plot_ibm_spectrum(experiment_path, save_path=save_path)
    return fig
