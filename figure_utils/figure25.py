"""
Figure 25: UMAP Visualization of Simulation and Real-World Feature Distributions.
Runs UMAP on Sivers input/output beam RSS features (sim vs. real-world) and
saves a 1×2 panel PDF.
"""

import os
import sys
import pickle
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from beamformer.dataset import data_process_4_weight_generator
from beamformer.weight_generator import ParametricGenerator
from beamformer.utils import get_uniform_samples, add_thermal_noise, read_csi_file_to_torch
from configs.submodules import dataset as cfg_dataset

REQUIRED_CONFIGS = []

_SIVERS_SIM_PATH = os.path.join(
    _PROJECT_ROOT,
    "csi-dataset/homeoffice-communication-28G-csi-sivers-indoor-patch/t4x4_r2x1_test_small",
)
_SIVERS_REAL_DIR = os.path.join(
    _PROJECT_ROOT,
    "csi-dataset/realworld_sivers/indoor-nlos",
)
_GENERATOR_PATH = os.path.join(
    _PROJECT_ROOT,
    "saved_models/sivers/generator_epoch_final.pth",
)
_MAX_SAMPLES = 800


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_sivers_real(max_samples=_MAX_SAMPLES):
    pkl_files = sorted(f for f in os.listdir(_SIVERS_REAL_DIR) if f.endswith('.pkl'))
    ml_list, sweep_list = [], []
    for fname in pkl_files[:max_samples]:
        with open(os.path.join(_SIVERS_REAL_DIR, fname), 'rb') as fp:
            data = pickle.load(fp)
        ml_list.append(np.array(data['ml_rss'])[8:24])    # 16 beams
        sweep_list.append(np.array(data['sweep_rss']))     # 200 beams
    return np.array(ml_list, dtype=np.float32), np.array(sweep_list, dtype=np.float32)


def _make_dp(ds):
    return data_process_4_weight_generator(
        device=torch.device('cpu'),
        M_tx=ds.M_tx, N_tx=ds.N_tx,
        M_rx=ds.M_rx, N_rx=ds.N_rx,
        mode=ds.mode,
        angle_steps_theta=20, angle_steps_phi=80,
        array_factor_steps_theta=16, array_factor_steps_phi=64,
        d_row=ds.d_row, d_col=ds.d_col,
        start_freq=ds.start_freq, end_freq=ds.end_freq,
        freq_num=ds.freq_num,
        max_theta=ds.max_theta,
    )


def _compute_rss_from_files(mat_files, dp, input_weights, output_weights,
                             ds, snr_db, max_samples):
    in_list, out_list = [], []
    for fpath in mat_files[:max_samples]:
        csi = read_csi_file_to_torch(
            fpath,
            M_tx=ds.M_tx, N_tx=ds.N_tx,
            M_rx=ds.M_rx, N_rx=ds.N_rx,
        )
        csi = add_thermal_noise(csi, subcarrier_bw=ds.subcarrier_spacing, snr_db=snr_db)
        csi = csi.unsqueeze(0)
        in_rss  = dp.generate_sample_rss(csi, input_weights).squeeze().detach().numpy()
        out_rss = dp.generate_sample_rss(csi, output_weights).squeeze().detach().numpy()
        in_list.append(in_rss.astype(np.float32))
        out_list.append(out_rss.astype(np.float32))
    return np.array(in_list), np.array(out_list)


def _load_sivers_sim(max_samples=_MAX_SAMPLES):
    ds = cfg_dataset.sivers_indoor(add_noise=False, snr_min=None)
    dp = _make_dp(ds)

    param_gen = ParametricGenerator(channel_number=16, M_base=ds.M, N_base=ds.N)
    param_gen.load_state_dict(torch.load(_GENERATOR_PATH, map_location='cpu', weights_only=True))
    param_gen.eval()
    with torch.no_grad():
        in_weights, _ = param_gen.generate(hardware=True)  # [1, 16, 4, 4]

    ant_info = dp.antenna_info
    sample_points = get_uniform_samples(200, 40)
    out_w = torch.zeros(ds.M, ds.N, 200, dtype=torch.complex64)
    for i, (phi, theta) in enumerate(sample_points):
        w = ant_info.generate_toward_angles([theta], [phi])
        out_w[:, :, i] = torch.from_numpy(w).reshape(ds.M, ds.N)
    out_weights = out_w.permute(2, 0, 1).unsqueeze(0)  # [1, 200, 4, 4]

    mat_files = sorted(
        os.path.join(_SIVERS_SIM_PATH, f)
        for f in os.listdir(_SIVERS_SIM_PATH)
        if f.endswith('.mat')
    )
    return _compute_rss_from_files(mat_files, dp, in_weights, out_weights,
                                   ds, snr_db=-3, max_samples=max_samples)


# ── Normalise ──────────────────────────────────────────────────────────────────

def _normalise(arr):
    mx = arr.max(axis=1, keepdims=True)
    mx = np.where(mx == 0, 1.0, mx)
    return arr / mx


# ── UMAP 1×2 grid ──────────────────────────────────────────────────────────────

def _run_umap_grid(panels, save_path):
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype']  = 42

    fig, axes = plt.subplots(1, 2, figsize=(6.66, 3.0))

    for ax, panel in zip(axes.flat, panels):
        sim_feat  = panel['sim_feat']
        real_feat = panel['real_feat']
        title     = panel['title']

        n_sim    = len(sim_feat)
        all_feat = np.vstack([sim_feat, real_feat])

        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            embedding = reducer.fit_transform(all_feat)

        sim_emb  = embedding[:n_sim]
        real_emb = embedding[n_sim:]

        ax.scatter(sim_emb[:, 0],  sim_emb[:, 1],  c='#5B8DB8', s=4,  alpha=0.35,
                   linewidths=0, label='Simulation', rasterized=True)
        ax.scatter(real_emb[:, 0], real_emb[:, 1], c='#E8604C', s=10, alpha=0.90,
                   linewidths=0, marker='s', label='Real-world', rasterized=True)

        ax.set_title(title, fontsize=12, fontweight='bold', fontname='Arial', pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles, labels = axes.flat[-1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=2,
                        fontsize=12, frameon=False,
                        markerscale=3, bbox_to_anchor=(0.5, -0.04))
    for text in legend.get_texts():
        text.set_fontname('Arial')

    fig.tight_layout(pad=0.5, rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


# ── Public interface ───────────────────────────────────────────────────────────

def plot(save_path):
    """
    Plot Figure 25: UMAP visualization of simulation vs. real-world feature distributions.

    Loads Sivers realworld pkl files and simulation .mat files, computes beam RSS
    features for input (16-beam) and output (200-beam), normalises per sample, then
    runs UMAP and saves a 1×2 panel PDF.

    Args:
        save_path: output path for the figure (extension forced to .pdf)
    """
    save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
    os.makedirs(os.path.dirname(os.path.abspath(save_path_pdf)), exist_ok=True)

    print('[Figure 25] Loading Sivers realworld data...')
    sv_real_in, sv_real_out = _load_sivers_real()
    print(f'  real:  input {sv_real_in.shape}  output {sv_real_out.shape}')

    print('[Figure 25] Loading Sivers simulation data...')
    sv_sim_in, sv_sim_out = _load_sivers_sim()
    print(f'  sim:   input {sv_sim_in.shape}  output {sv_sim_out.shape}')

    sv_sim_in_n   = _normalise(sv_sim_in)
    sv_real_in_n  = _normalise(sv_real_in)
    sv_sim_out_n  = _normalise(sv_sim_out)
    sv_real_out_n = _normalise(sv_real_out)

    print('[Figure 25] Running UMAP...')
    _run_umap_grid(
        panels=[
            dict(sim_feat=sv_sim_in_n,  real_feat=sv_real_in_n,
                 title='Sivers Input Feature'),
            dict(sim_feat=sv_sim_out_n, real_feat=sv_real_out_n,
                 title='Sivers Output Feature'),
        ],
        save_path=save_path_pdf,
    )
    print(f'[Figure 25] Saved → {save_path_pdf}')
