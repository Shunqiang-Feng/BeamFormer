"""
cdf_plot.py — CDF comparison plot: our model vs baselines.

Our model is evaluated live on the test dataset. Baseline data is loaded from
baselines_result/all_performance_dict.json (published alongside the code).

Output: figures/cdf_comparison.png

Usage:
    python beamformer/cdf_plot.py
    python beamformer/cdf_plot.py --data_path csi-dataset/homeoffice-communication-28G-csi/t16x16_r2x1_test_small
    python beamformer/cdf_plot.py --baselines_json baselines_result/all_performance_dict.json
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from beamformer.utils import read_csi_file_to_torch, gpu_tensor_to_np, get_db
from beamformer.inference_helper import InferenceHelper

FIGURES_DIR = "figures"
DEFAULT_DATA_PATH = "mini_demo/indoor_28g_dataset/t16x16_r2x1_test_small"
DEFAULT_MODEL_DIR = "mini_demo/saved_models"
DEFAULT_ARN_MODEL_DIR = "mini_demo/saved_models"
DEFAULT_BASELINES_JSON = "baselines_result/all_performance_dict.json"

BASELINE_KEYS = ["AgileLink", "SectorSweep", "Hierarchical", "802ad", "2ACE", "MLP", "CNN"]


def build_setting(data_path, model_dir=DEFAULT_MODEL_DIR, arn_model_dir=DEFAULT_ARN_MODEL_DIR):
    from types import SimpleNamespace
    from configs.submodules import assumption, dataset, estimator, generator, ARN_model

    ds = dataset.homeoffice_communication_28g()
    ds.test_data_path = data_path

    setting = SimpleNamespace(
        name="cdf_eval",
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


def compute_max_rss_db(dp, csi_tensor):
    """Compute optimal (oracle) RSS for a CSI sample via exhaustive beam sweep."""
    csi = csi_tensor.to(dp.device)
    max_rss = dp.generate_max_rss(csi)   # [1]
    return get_db(max_rss[0].item())


def run_our_model(helper, data_path):
    """Evaluate our model on all CSI files and return list of RSS loss values (dB)."""
    files = sorted([
        os.path.join(data_path, f)
        for f in os.listdir(data_path) if f.endswith(".mat")
    ])
    if not files:
        raise FileNotFoundError(f"No .mat files found in {data_path}")

    losses = []
    print(f"[CDF] Evaluating our model on {len(files)} CSI files ...")
    for csi_path in tqdm(files):
        csi_tensor = read_csi_file_to_torch(csi_path).unsqueeze(0)

        max_rss_db = compute_max_rss_db(helper.dp, csi_tensor)

        sample_rss, query_rss, query_rss_pred, scale, _ = helper.infer_from_csi(csi_tensor)
        pred_spectrum = helper.apply_arn(sample_rss, query_rss_pred, scale)

        peak_idx = np.unravel_index(np.argmax(pred_spectrum), pred_spectrum.shape)
        gt_spectrum = gpu_tensor_to_np(query_rss.reshape(80, 20)) * scale
        rss_at_pred_peak_db = get_db(gt_spectrum[peak_idx])

        losses.append(max_rss_db - rss_at_pred_peak_db)

    return losses


NAME_SWITCH_TABLE = {
    'Our Method': 'BeamFormer',
    '802ad': 'Fine Sweep',
    'SectorSweep': 'Coarse Sweep',
    'Hierarchical': 'Hier. Sweep',
    '2ACE': '2ACE',
    'AgileLink': 'AgileLink',
    'MLP': 'MLP',
    'CNN': 'CNN',
}


def display_name(name):
    return NAME_SWITCH_TABLE.get(name, name)


def load_baseline_losses(json_path, keys):
    """Load pre-computed loss values for baselines from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    result = {}
    for key in keys:
        if key not in data:
            print(f"[CDF] Warning: '{key}' not found in {json_path}, skipping.")
            continue
        result[key] = [entry["loss"] for entry in data[key]]
    return result


def plot_cdf(losses_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    basic_fontsize = 16
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    for i, (method, losses) in enumerate(losses_dict.items()):
        sorted_losses = np.sort(losses)
        cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        markevery = max(1, len(sorted_losses) // 10)
        ax.plot(sorted_losses, cdf,
                label=display_name(method),
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markevery=markevery,
                markersize=4,
                linewidth=2,
                alpha=0.8)

    ax.set_xlabel("RSS Loss (dB)", fontsize=basic_fontsize)
    ax.set_ylabel("CDF", fontsize=basic_fontsize)
    ax.legend(fontsize=basic_fontsize - 2, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=basic_fontsize - 2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[CDF] Saved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="CDF comparison: our model vs baselines")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--arn_model_dir", type=str, default=DEFAULT_ARN_MODEL_DIR)
    parser.add_argument("--baselines_json", type=str, default=DEFAULT_BASELINES_JSON)
    parser.add_argument("--output", type=str,
                        default=os.path.join(FIGURES_DIR, "cdf_comparison.png"))
    args = parser.parse_args()

    setting = build_setting(args.data_path, args.model_dir, args.arn_model_dir)
    helper = InferenceHelper(setting)

    our_losses = run_our_model(helper, args.data_path)
    baseline_losses = load_baseline_losses(args.baselines_json, BASELINE_KEYS)

    all_losses = {"Our Method": our_losses}
    all_losses.update(baseline_losses)

    plot_cdf(all_losses, args.output)


if __name__ == "__main__":
    main()
