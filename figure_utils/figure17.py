"""
Figure 17: Impact of latency and user mobility.
For each CSI sample the model predicts a beam direction using tw=0ms CSI, then
evaluates RSS loss at that direction on CSI collected at different latencies
(tw=1..10ms) and speeds (10/30/50/70 mph). Plots Mean RSS Loss vs Latency,
one curve per speed.
"""

import os
import sys
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REQUIRED_CONFIGS = []  # special case - uses dedicated velocity_test runner

_DEFAULT_DATA_PATH = os.path.join(
    _PROJECT_ROOT,
    "csi-dataset/mobility_dataset/t16x16_r2x1_test_small"
)
_DEFAULT_CSV = os.path.join(
    _PROJECT_ROOT,
    "eval_results", "velocity_test", "results.csv"
)

SPEED_ORDER = [10, 30, 50, 70, 90]
COLORS      = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
LINESTYLES  = ['-', '--', '-.', ':', '-']
MARKERS     = ['o', 's', '^', 'D', 'v']


# ── Runner ────────────────────────────────────────────────────────────────────

_FNAME_RE = re.compile(r'^speed(\d+)mph_tw(\d+)ms-(\d+)\.mat$')


def _parse_filename(fname):
    m = _FNAME_RE.match(os.path.basename(fname))
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _group_files(data_path):
    groups = defaultdict(dict)
    for fname in os.listdir(data_path):
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        speed, tw, idx = parsed
        groups[(speed, idx)][tw] = os.path.join(data_path, fname)
    return groups


def run(data_path=None, save_csv=None):
    """
    Run velocity robustness evaluation.

    For each sample, predicts direction from tw=0ms CSI, then evaluates RSS
    loss at that direction for every available time window.

    Args:
        data_path: path to mobility dataset directory
        save_csv: where to save the results CSV

    Returns:
        str: path to the saved CSV file
    """
    from scipy.constants import c as light_speed
    import torch
    from types import SimpleNamespace

    from beamformer.dataset import load_data_process
    from beamformer.modules import AmplitudeRecoveryNetwork, FastTransformerModel
    from beamformer.weight_generator import ParametricGenerator, transform_weights
    from beamformer.utils import read_csi_file_to_torch, gpu_tensor_to_np, get_db
    from configs.submodules import assumption, ARN_model, estimator as est_cfg_mod

    if data_path is None:
        data_path = _DEFAULT_DATA_PATH
    if save_csv is None:
        save_csv = _DEFAULT_CSV

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Mobility dataset not found: {data_path}\n"
            "Expected at: csi-dataset/mobility_dataset"
        )

    mid_freq = 27.925e9
    ds = SimpleNamespace(
        name='velocity_communication_28g',
        train_data_path=data_path,
        test_data_path=data_path,
        mode='rx_act1',
        freq_num=128,
        start_freq=27.90964e9,
        end_freq=27.94012e9,
        M_tx=16, N_tx=16,
        M_rx=2, N_rx=1,
        max_theta=90,
        add_noise=False,
        snr_min=None,
    )
    ds.M = ds.M_tx
    ds.N = ds.N_tx
    ds.d_row = 0.5 * light_speed / mid_freq
    ds.d_col = 0.5 * light_speed / mid_freq
    ds.subcarrier_spacing = (ds.end_freq - ds.start_freq) / (ds.freq_num - 1)

    asmp   = assumption.beam64_hr(phi_endpoint=False)
    config = SimpleNamespace(dataset=ds, assumption=asmp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Figure 17] Device: {device}')

    dp = load_data_process(config, device)

    # Generator
    gen = ParametricGenerator(asmp.sample_num, ds.M, ds.N)
    gen.load_state_dict(torch.load(
        'saved_models/ct_xa16/generator_epoch_final.pth',
        map_location=device, weights_only=True))
    gen.eval().to(device)

    # Estimator
    est_config = est_cfg_mod.PerceiverIO(
        depth=4, dim=512, latent_dim=512,
        estimator_pretrained_model='saved_models/ct_xa16/estimator_epoch_final.pth',
    )
    est = FastTransformerModel(est_config)
    est.load_state_dict(torch.load(
        est_config.estimator_pretrained_model,
        map_location=device, weights_only=True))
    est.eval().to(device)

    # ARN (no pretrained)
    arn_cfg = ARN_model.typical_ARN(None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        arn = AmplitudeRecoveryNetwork(
            sample_num=asmp.sample_num, d_model=arn_cfg.hidden_dims)
    arn.has_pretrained = False
    arn.eval().to('cpu')

    groups     = _group_files(data_path)
    valid_items = [(k, v) for k, v in sorted(groups.items()) if 0 in v]
    total = len(valid_items)
    print(f'[Figure 17] Total samples with tw=0: {total}')

    records = []

    for i, ((speed, idx), tw_dict) in enumerate(valid_items):
        print(f'\r  [{i+1}/{total}] speed={speed}mph, sample={idx}', end='', flush=True)

        csi0 = read_csi_file_to_torch(
            tw_dict[0],
            frequency_num=ds.freq_num,
            M_tx=ds.M_tx, N_tx=ds.N_tx,
            M_rx=ds.M_rx, N_rx=ds.N_rx,
        ).unsqueeze(0)

        csi0 = csi0.to(device)
        with torch.no_grad():
            z = torch.randn(1, gen.channel_number * dp.M * dp.N).to(device)
            raw_weights = gen(z)
            weights_out, _ = transform_weights(raw_weights)

            sample_pos_enc = dp.generate_sample_position_encoding(weights_out).to(dtype=torch.float32)
            query_pos_enc  = dp.generate_query_position_encoding(batch_size=1).to(dtype=torch.float32)
            sample_rss = dp.generate_sample_rss(csi0, weights_out).to(dtype=torch.float32)
            scale = torch.max(sample_rss, dim=1, keepdim=True).values
            sample_rss_norm = sample_rss / scale

            spe, qpe = est.prepare_positional_encoding(sample_pos_enc, query_pos_enc)
            query_rss_pred, _ = est(sample_rss_norm, spe, qpe)

        phi_steps   = dp.angle_steps_phi
        theta_steps = dp.angle_steps_theta
        pred_spec_np = gpu_tensor_to_np(query_rss_pred[0].reshape(phi_steps, theta_steps))
        phi_idx, theta_idx = np.unravel_index(np.argmax(pred_spec_np), pred_spec_np.shape)
        phi_pred   = float(phi_idx)
        theta_pred = float(theta_idx)

        for tw, fpath in tw_dict.items():
            csi_tw = read_csi_file_to_torch(
                fpath,
                frequency_num=ds.freq_num,
                M_tx=ds.M_tx, N_tx=ds.N_tx,
                M_rx=ds.M_rx, N_rx=ds.N_rx,
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                max_rss_val, _, _, angle_spectrum = dp.generate_max_rss(csi_tw, return_direction=True)

            spec_np = angle_spectrum[0].cpu().numpy()
            max_theta_val = dp.antenna_info.max_theta
            phi_values   = np.arange(spec_np.shape[0]) * (360.0 / spec_np.shape[0])
            theta_values = np.arange(spec_np.shape[1]) * (max_theta_val / spec_np.shape[1])
            p_idx = int(np.argmin(np.abs(phi_values   - phi_pred)))
            t_idx = int(np.argmin(np.abs(theta_values - theta_pred)))
            rss_at_dir = float(spec_np[p_idx, t_idx])
            max_rss    = float(max_rss_val[0].cpu())

            rss_at_dir_db = get_db(rss_at_dir)
            max_rss_db    = get_db(max_rss)
            rss_loss_db   = max_rss_db - rss_at_dir_db

            records.append({
                'speed_mph':      speed,
                'tw_ms':          tw,
                'sample_idx':     idx,
                'phi_pred':       phi_pred,
                'theta_pred':     theta_pred,
                'rss_at_pred_db': rss_at_dir_db,
                'max_rss_db':     max_rss_db,
                'rss_loss_db':    rss_loss_db,
            })

    print()
    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
    df.to_csv(save_csv, index=False, float_format='%.6f')
    print(f'[Figure 17] Results saved to {save_csv}')

    return save_csv


def load_from_csv(csv_path=None):
    """
    Load pre-computed velocity test results.

    Args:
        csv_path: path to CSV produced by run()

    Returns:
        pd.DataFrame
    """
    if csv_path is None:
        csv_path = _DEFAULT_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Velocity results CSV not found: {csv_path}\n"
            "Run with --data_source from_scratch to generate results."
        )
    df = pd.read_csv(csv_path)
    print(f"[Figure 17] Loaded {len(df)} records from {csv_path}")
    return df


def plot(data, save_path):
    """
    Plot Figure 17: Impact of latency and user mobility.

    Args:
        data: pd.DataFrame from load_from_csv(), or a str CSV path
        save_path: output figure path
    """
    if isinstance(data, str):
        data = load_from_csv(data)

    tw_vals = sorted(data["tw_ms"].unique())

    basic_fontsize = 16
    plot_width  = 3.33
    plot_height = 3.33 * 0.85
    left_margin, right_margin = 0.8, 0.3
    bottom_margin, top_margin = 0.7, 0.5

    fig_width  = left_margin + plot_width + right_margin
    fig_height = bottom_margin + plot_height + top_margin

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    ax  = fig.add_axes([
        left_margin / fig_width,
        bottom_margin / fig_height,
        plot_width / fig_width,
        plot_height / fig_height,
    ])

    for i, speed in enumerate(SPEED_ORDER):
        sub = data[data["speed_mph"] == speed]
        if sub.empty:
            continue
        means = [sub.loc[sub["tw_ms"] == tw, "rss_loss_db"].mean() for tw in tw_vals]
        ax.plot(
            tw_vals, means,
            label=f"{speed} mph",
            color=COLORS[i % len(COLORS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=5,
            linewidth=2,
            alpha=0.85,
        )

    ax.set_xlabel("Latency (ms)", fontsize=basic_fontsize)
    ax.set_ylabel("RSS Loss (dB)", fontsize=basic_fontsize)
    ax.tick_params(axis='both', labelsize=basic_fontsize - 2)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=basic_fontsize - 2)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    print(f"[Figure 17] Figure saved to {save_path}")
    plt.close()
