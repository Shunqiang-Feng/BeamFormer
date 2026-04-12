"""
Figure 13: Failure case under pure noise.
Beam spectrum visualization (GT=zeros, Pred) showing model output under pure-noise input.
Uses config: visualize_ct_xa16_noise (homeoffice_communication_28g with SNR=-100, noise_gt=True).
"""
import os
import sys
import glob
import shutil

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REQUIRED_CONFIGS = []
CONFIG_NAME = "visualize_ct_xa16_noise"
_MODEL_NAME = "ct_xa16"


def plot(save_path):
    """
    Run beam spectrum visualization for Figure 13 (pure-noise failure case).

    Triggers the evaluator in visualize mode using config visualize_ct_xa16_noise.
    GT is replaced with zeros (noise_gt=True in config) to represent pure noise input.
    The GT & Pred polar disk images are saved to eval_results/visualize_ct_xa16/<model>/
    and the most recent one is copied to save_path (as .png).

    Args:
        save_path: base path for the output figure (extension forced to .png)
    """
    from beamformer.utils import load_config
    from beamformer.evaluator import Evaluator

    print(f"[Figure 13] Loading config: {CONFIG_NAME}")
    config = load_config(CONFIG_NAME, predix="evaluate")

    TARGET = "seed1320-epoch28-56.mat"
    evaluator = Evaluator(config)
    evaluator.visualize_one_file(TARGET)

    results_dir = config.request.results_folder
    model_dir = os.path.join(results_dir, _MODEL_NAME)
    png_files = sorted(glob.glob(os.path.join(model_dir, "*.png")))
    if not png_files:
        raise FileNotFoundError(
            f"No PNG files found in {model_dir}\n"
            "Ensure the model weights exist at saved_models/ct_xa16/ and the dataset is available."
        )

    save_path_png = os.path.splitext(save_path)[0] + '.png'
    os.makedirs(os.path.dirname(os.path.abspath(save_path_png)), exist_ok=True)
    shutil.copy2(png_files[-1], save_path_png)
    print(f"[Figure 13] Spectrum visualization saved to: {save_path_png}")
    print(f"[Figure 13] All visualizations in: {model_dir}/")
    return save_path_png
