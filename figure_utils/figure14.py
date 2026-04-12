"""
Figure 14: Beam Spectrum Resolution.
Generates a 2×4 polar disk figure for beam_res evaluation.
Top row   : 4 GT spectra
Bottom row: 4 Pred spectra
Order (left→right): csi_K100_0040, csi_K100_0060, csi_K100_0080, csi_K100_0100
"""

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from beamformer.dataset import load_datasets, load_data_process
from beamformer.modules import AmplitudeRecoveryNetwork, FastTransformerModel
from beamformer.weight_generator import ParametricGenerator, transform_weights
from beamformer.utils import gpu_tensor_to_np, _tensor_to_disk_points, _make_disk_grid, _draw_polar_disk_subplot
from configs.evaluate.test_beam_res import config

REQUIRED_CONFIGS = []

TARGET_FILES = [
    "csi_K100_0040.mat",
    "csi_K100_0060.mat",
    "csi_K100_0080.mat",
    "csi_K100_0100.mat",
]
FIG_WIDTH = 6.66     # inches
ROW_GAP   = 0.1      # inches between GT and Pred rows
COLORMAP  = "viridis"


# ── Model initialisation ───────────────────────────────────────────────────────

def _init_models(setting, device):
    ds         = setting.dataset
    sample_num = setting.assumption.sample_num

    generator = ParametricGenerator(sample_num, ds.M, ds.N)
    generator.load_state_dict(torch.load(
        setting.generator.generator_pretrained_model,
        map_location=device, weights_only=True,
    ))
    generator.eval().to(device)

    estimator = FastTransformerModel(setting.estimator)
    estimator.load_state_dict(torch.load(
        setting.estimator.estimator_pretrained_model,
        map_location=device, weights_only=True,
    ))
    estimator.eval().to(device)

    arn_cfg = setting.arn_model
    arn = AmplitudeRecoveryNetwork(sample_num=sample_num, d_model=arn_cfg.hidden_dims)
    arn.has_pretrained = False
    if arn_cfg.ARN_model_pretrained_model is not None:
        arn.load_state_dict(torch.load(
            arn_cfg.ARN_model_pretrained_model,
            map_location=device, weights_only=True,
        ))
        arn.has_pretrained = True
    arn.eval().to('cpu')

    return generator, estimator, arn


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(setting, csi_tensor, generator, estimator, arn, dp, device):
    """Return (gt_2d, pred_2d) numpy arrays, shape (phi_steps, theta_steps)."""
    phi_steps   = setting.assumption.angle_steps_phi
    theta_steps = setting.assumption.angle_steps_theta
    ds          = setting.dataset

    csi        = csi_tensor.to(device)
    batch_size = csi.shape[0]

    z           = torch.randn(batch_size, setting.assumption.sample_num * ds.M * ds.N).to(device)
    raw_weights = generator(z)
    weights_out, _ = transform_weights(raw_weights)

    sample_pos_enc = dp.generate_sample_position_encoding(weights_out).to(torch.float32)
    query_pos_enc  = dp.generate_query_position_encoding(batch_size=batch_size).to(torch.float32)
    sample_rss     = dp.generate_sample_rss(csi, weights_out).to(torch.float32)
    query_rss      = dp.generate_query_rss(csi).to(torch.float32)
    scale          = torch.max(sample_rss, dim=1, keepdim=True).values
    sample_rss    /= scale
    query_rss     /= scale

    sample_enc, query_enc = estimator.prepare_positional_encoding(sample_pos_enc, query_pos_enc)
    pred, _ = estimator(sample_rss, sample_enc, query_enc, None)

    if arn.has_pretrained:
        pred_lobe = 1.0 / torch.mean(arn(sample_rss.cpu().unsqueeze(0))).item()
    else:
        pred_lobe = 1.0

    gt_2d   = gpu_tensor_to_np(query_rss[0].reshape(phi_steps, theta_steps))
    pred_2d = gpu_tensor_to_np(pred[0].reshape(phi_steps, theta_steps)) * pred_lobe
    return gt_2d, pred_2d


# ── Draw a single disk (no labels / colorbar) ──────────────────────────────────

def _draw_disk(ax, tensor_np, max_theta, x_grid, y_grid, mask, r_max, cmap):
    x_pts, y_pts, values = _tensor_to_disk_points(tensor_np, max_theta)
    _draw_polar_disk_subplot(
        ax, x_pts, y_pts, values,
        x_grid, y_grid, mask, r_max, cmap,
        title='',
        show_colorbar=False,
        show_angle_labels=False,
        peak_xy=None,
    )
    ax.set_title('')


# ── Public interface ───────────────────────────────────────────────────────────

def plot(save_path):
    """
    Generate a 2×4 polar disk figure for beam_res evaluation.

    Loads the ct_xa16 model from configs/evaluate/test_beam_res.py, looks up the
    4 TARGET_FILES in the test dataset, runs inference, and saves a combined figure
    with GT spectra on top and Pred spectra on the bottom.

    Args:
        save_path: output path (extension forced to .png)
    """
    setting   = config.models[0]
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_theta = setting.dataset.max_theta

    print('[Figure 14] Loading models...')
    generator, estimator, arn = _init_models(setting, device)
    dp = load_data_process(setting, device)

    _, test_dataset = load_datasets(setting)
    path_to_idx = {
        os.path.basename(test_dataset.dataset_path[i]): i
        for i in range(len(test_dataset))
    }

    print('[Figure 14] Running inference...')
    gt_list, pred_list = [], []
    for fname in TARGET_FILES:
        if fname not in path_to_idx:
            raise FileNotFoundError(f"{fname} not found in test dataset")
        csi_tensor = test_dataset[path_to_idx[fname]][0].unsqueeze(0)
        gt, pred = _run_inference(setting, csi_tensor, generator, estimator, arn, dp, device)
        gt_list.append(gt)
        pred_list.append(pred)
        print(f'  {fname} ✓')

    # ── Build 2×4 figure ──────────────────────────────────────────────────────
    n_cols     = 4
    subplot_w  = FIG_WIDTH / n_cols
    fig_height = 2 * subplot_w + ROW_GAP
    hspace     = ROW_GAP / subplot_w

    fig = plt.figure(figsize=(FIG_WIDTH, fig_height), facecolor='white')
    gs  = gridspec.GridSpec(
        2, n_cols,
        figure=fig,
        left=0, right=1, top=1, bottom=0,
        wspace=0, hspace=hspace,
    )

    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(color='none')
    x_grid, y_grid, mask, r_max = _make_disk_grid(max_theta)

    for col in range(n_cols):
        _draw_disk(fig.add_subplot(gs[0, col]), gt_list[col],
                   max_theta, x_grid, y_grid, mask, r_max, cmap)
        _draw_disk(fig.add_subplot(gs[1, col]), pred_list[col],
                   max_theta, x_grid, y_grid, mask, r_max, cmap)

    save_path_png = os.path.splitext(save_path)[0] + '.png'
    os.makedirs(os.path.dirname(os.path.abspath(save_path_png)), exist_ok=True)
    fig.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[Figure 14] Saved → {save_path_png}')
