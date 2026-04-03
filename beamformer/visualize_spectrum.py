"""
visualize_spectrum.py — Randomly sample 8x4 CSI files and plot their angle spectra.

Each sample produces a GT | Prediction pair rendered as polar disk plots,
laid out in an 8x4 grid (32 pairs total).
Output: figures/spectrum_grid.png

Usage:
    python beamformer/visualize_spectrum.py
    python beamformer/visualize_spectrum.py --data_path csi-dataset/homeoffice-communication-28G-csi/t16x16_r2x1_test_small
    python beamformer/visualize_spectrum.py --rows 4 --cols 4 --seed 0
"""

import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from beamformer.utils import (
    read_csi_file_to_torch, gpu_tensor_to_np,
    _make_disk_grid, _tensor_to_disk_points, _draw_polar_disk_subplot,
)
from beamformer.inference_helper import InferenceHelper

FIGURES_DIR = "figures"
DEFAULT_DATA_PATH = "mini_demo/indoor_28g_dataset/t16x16_r2x1_test_small"
DEFAULT_MODEL_DIR = "mini_demo/saved_models"
DEFAULT_ARN_MODEL_DIR = "mini_demo/saved_models"
MAX_THETA = 90


def build_setting(data_path, model_dir=DEFAULT_MODEL_DIR, arn_model_dir=DEFAULT_ARN_MODEL_DIR):
    from types import SimpleNamespace
    from configs.submodules import assumption, dataset, estimator, generator, ARN_model

    ds = dataset.homeoffice_communication_28g()
    ds.test_data_path = data_path

    setting = SimpleNamespace(
        name="visualize",
        dataset=ds,
        assumption=assumption.beam64(),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model=os.path.join(model_dir, "generator.pth")
        ),
        estimator=estimator.PerceiverIO(
            estimator_pretrained_model=os.path.join(model_dir, "estimator.pth"),
        ),
        arn_model=ARN_model.typical_ARN(
            ARN_model_pretrained_model=os.path.join(arn_model_dir, "arn_model.pth")
        ),
    )
    return setting


def main():
    parser = argparse.ArgumentParser(description="Visualize angle spectra for sampled CSI files")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--arn_model_dir", type=str, default=DEFAULT_ARN_MODEL_DIR)
    parser.add_argument("--rows", type=int, default=8, help="Number of sample rows")
    parser.add_argument("--cols", type=int, default=4, help="Number of sample columns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=os.path.join(FIGURES_DIR, "spectrum_grid.png"))
    args = parser.parse_args()

    random.seed(args.seed)
    n_samples = args.rows * args.cols

    setting = build_setting(args.data_path, args.model_dir, args.arn_model_dir)
    helper = InferenceHelper(setting)

    all_files = sorted([
        os.path.join(args.data_path, f)
        for f in os.listdir(args.data_path) if f.endswith(".mat")
    ])
    if not all_files:
        raise FileNotFoundError(f"No .mat files found in {args.data_path}")

    selected = random.sample(all_files, min(n_samples, len(all_files)))
    print(f"[Visualize] Plotting {len(selected)} samples in a {args.rows}×{args.cols} grid")

    # Build shared disk grid once (coarser spacing for grid performance)
    x_grid, y_grid, mask, r_max = _make_disk_grid(MAX_THETA, spacing=0.005)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='none')

    # Each sample = 2 subplots (GT | Pred); total grid: rows × (cols*2)
    fig_cols = args.cols * 2
    fig, axes = plt.subplots(args.rows, fig_cols,
                              figsize=(fig_cols * 3, args.rows * 3))
    axes = np.array(axes).reshape(args.rows, fig_cols)

    for idx, csi_path in enumerate(selected):
        row = idx // args.cols
        col_base = (idx % args.cols) * 2

        csi_tensor = read_csi_file_to_torch(csi_path).unsqueeze(0)
        sample_rss, query_rss, query_rss_pred, scale, _ = helper.infer_from_csi(csi_tensor)
        pred_spectrum = helper.apply_arn(sample_rss, query_rss_pred, scale)
        gt_spectrum = gpu_tensor_to_np(query_rss.reshape(80, 20)) * scale

        gt_x, gt_y, gt_vals = _tensor_to_disk_points(gt_spectrum, MAX_THETA)
        pred_x, pred_y, pred_vals = _tensor_to_disk_points(pred_spectrum, MAX_THETA)

        _draw_polar_disk_subplot(
            axes[row, col_base],
            gt_x, gt_y, gt_vals,
            x_grid, y_grid, mask, r_max, cmap,
            title=f"GT #{idx + 1}",
        )
        _draw_polar_disk_subplot(
            axes[row, col_base + 1],
            pred_x, pred_y, pred_vals,
            x_grid, y_grid, mask, r_max, cmap,
            title=f"Pred #{idx + 1}",
        )

        print(f"  [{idx+1}/{len(selected)}] {os.path.basename(csi_path)}")

    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved to {args.output}")


if __name__ == "__main__":
    main()
