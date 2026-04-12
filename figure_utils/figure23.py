"""
Figure 23: Example spectrum with Sivers SDR.
Uses compare_scan_with_pred on csi-dataset/realworld_sivers/indoor-nlos/1.pkl.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REQUIRED_CONFIGS = []

# Default data path for Figure 23
DEFAULT_PKL_PATH = os.path.join(
    _PROJECT_ROOT, "csi-dataset/realworld_sivers/indoor-nlos/1.pkl"
)


def plot(save_path, pkl_path=None):
    """
    Plot Figure 23: Sivers SDR spectrum example.

    Args:
        save_path: path to save the figure
        pkl_path: path to sivers pkl file (default: csi-dataset/realworld_sivers/indoor-nlos/1.pkl)
    """
    from beamformer.hardware_inference import plot_sivers_spectrum

    if pkl_path is None:
        pkl_path = DEFAULT_PKL_PATH

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Sivers data not found: {pkl_path}\n"
            f"Expected at: csi-dataset/realworld_sivers/indoor-nlos/1.pkl"
        )

    print(f"[Figure 23] Generating Sivers spectrum from: {pkl_path}")
    fig = plot_sivers_spectrum(pkl_path, save_path=save_path)
    return fig
