"""
Figure 15: Multi Path Prediction Accuracy.
CDF of RSS prediction error for the 1st, 2nd, and 3rd strongest paths.
Evaluation uses the ct_xa16_large_ft model on homeoffice 28G 16x16 indoor data.
"""

import os
import sys
import json

import torch
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from figure_utils.shared_utils import plot_multi_cdf

REQUIRED_CONFIGS = []  # special case - uses dedicated multi-path runner

_DEFAULT_RESULTS_PATH = os.path.join(
    _PROJECT_ROOT, "eval_results", "multi_path_eval_new", "results.json"
)

NUM_PATHS   = 3
MIN_SEP_DEG = 5.0   # minimum angular separation between selected paths


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_metainfo_dir(test_data_path):
    path   = test_data_path.rstrip("/\\")
    parent = os.path.dirname(path)
    return os.path.join(parent, "t16x16_r2x1_metainfo_detail")


def _load_path_info(csi_path, metainfo_dir):
    basename = os.path.splitext(os.path.basename(csi_path))[0]
    csv_path = os.path.join(metainfo_dir, basename + ".csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Angular geometry
# ---------------------------------------------------------------------------

def _dir_to_cartesian(phi_deg, theta_deg):
    phi_r   = np.radians(phi_deg)
    theta_r = np.radians(theta_deg)
    return np.array([
        np.sin(theta_r) * np.cos(phi_r),
        np.sin(theta_r) * np.sin(phi_r),
        np.cos(theta_r),
    ])


def _angular_sep_deg(phi1, theta1, phi2, theta2):
    v1 = _dir_to_cartesian(phi1, theta1)
    v2 = _dir_to_cartesian(phi2, theta2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


# ---------------------------------------------------------------------------
# GT RSS via direct beamforming
# ---------------------------------------------------------------------------

def _gt_rss_at_dirs(csi_1, phi_arr, theta_arr, dp, device):
    n      = len(phi_arr)
    sv     = dp.antenna_info.generate_toward_angles(theta_arr, phi_arr)
    sv     = sv.reshape(dp.M, dp.N, n, n)
    sv_diag = sv[:, :, np.arange(n), np.arange(n)]                    # [M, N, n]
    weights = (
        torch.from_numpy(sv_diag)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=csi_1.dtype)
    )
    weights_flat = weights.view(1, n, -1)
    with torch.no_grad():
        rss = dp._generate_rss(csi_1, weights_flat)                    # [1, n]
    return rss[0].cpu().numpy().astype(np.float64)


def _select_top3_with_separation(csi_1, phi_arr, theta_arr, dp, device):
    gt_rss = _gt_rss_at_dirs(csi_1, phi_arr, theta_arr, dp, device)
    order  = np.argsort(gt_rss)[::-1]
    selected = []
    for idx in order:
        phi   = float(phi_arr[idx])
        theta = float(theta_arr[idx])
        rss   = float(gt_rss[idx])
        if all(
            _angular_sep_deg(phi, theta, sp, st) >= MIN_SEP_DEG
            for sp, st, _ in selected
        ):
            selected.append((phi, theta, rss))
        if len(selected) == NUM_PATHS:
            break
    return selected if len(selected) == NUM_PATHS else None


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def _init_generator(setting, device):
    from beamformer.weight_generator import ParametricGenerator
    ds         = setting.dataset
    sample_num = setting.assumption.sample_num
    gen = ParametricGenerator(sample_num, ds.M, ds.N)
    if getattr(setting.generator, "generator_pretrained_model", None):
        gen.load_state_dict(
            torch.load(setting.generator.generator_pretrained_model,
                       map_location=device, weights_only=True)
        )
        print(f"Loaded generator from {setting.generator.generator_pretrained_model}")
    gen.eval().to(device)
    return gen


def _init_estimator(setting, device):
    from beamformer.modules import FastTransformerModel
    cfg = setting.estimator
    est = FastTransformerModel(cfg)
    est.load_state_dict(
        torch.load(cfg.estimator_pretrained_model, map_location=device, weights_only=True)
    )
    print(f"Loaded estimator from {cfg.estimator_pretrained_model}")
    est.eval().to(device)
    return est


def _init_arn(setting, device):
    from beamformer.modules import AmplitudeRecoveryNetwork
    cfg        = setting.arn_model
    sample_num = setting.assumption.sample_num
    arn        = AmplitudeRecoveryNetwork(sample_num=sample_num, d_model=cfg.hidden_dims)
    assert cfg.ARN_model_pretrained_model is not None, \
        "ARN pretrained model path must be provided for multi-path evaluation."
    arn.load_state_dict(
        torch.load(cfg.ARN_model_pretrained_model, map_location=device, weights_only=True)
    )
    print(f"Loaded ARN from {cfg.ARN_model_pretrained_model}")
    arn.eval().to("cpu")
    return arn


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(estimator, weights, csi, dp):
    sample_pos_enc  = dp.generate_sample_position_encoding(weights).to(dtype=torch.float32)
    query_pos_enc   = dp.generate_query_position_encoding(batch_size=csi.shape[0]).to(dtype=torch.float32)
    sample_rss      = dp.generate_sample_rss(csi, weights).to(dtype=torch.float32)
    scale           = torch.max(sample_rss, dim=1, keepdim=True).values
    sample_rss_norm = sample_rss / scale
    sp_enc, qp_enc  = estimator.prepare_positional_encoding(sample_pos_enc, query_pos_enc)
    query_rss_pred, _ = estimator(sample_rss_norm, sp_enc, qp_enc)
    return sample_rss_norm, scale, query_rss_pred


# ---------------------------------------------------------------------------
# RSS error
# ---------------------------------------------------------------------------

def _rss_error_arn(pred_2d_np, pred_lobe, scale_val, phi, theta, gt_rss_linear, max_theta):
    from beamformer.utils import find_closest_in_angle_spectrum
    val_at_dir  = float(find_closest_in_angle_spectrum(phi, theta, pred_2d_np, max_theta=max_theta))
    pred_linear = val_at_dir * pred_lobe * scale_val
    pred_db     = 10.0 * np.log10(max(pred_linear,   1e-30))
    gt_db       = 10.0 * np.log10(max(gt_rss_linear, 1e-30))
    return gt_db - pred_db


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _run_evaluation(num_samples=None, batch_size=1):
    """
    Run multi-path RSS error evaluation using ct_xa16_large_ft config.

    Returns:
        pd.DataFrame with columns path{k}_error_db, path{k}_phi, path{k}_theta,
        path{k}_gt_rss_db for k in 1..3
    """
    from beamformer.dataset import load_data_process, load_datasets
    from beamformer.weight_generator import transform_weights
    from configs.evaluate.test_ct_xa16_indoor_large import config

    setting = config.models[0]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metainfo_dir = _get_metainfo_dir(setting.dataset.test_data_path)
    print(f"Metainfo dir: {metainfo_dir}")

    generator  = _init_generator(setting, device)
    estimator  = _init_estimator(setting, device)
    arn_model  = _init_arn(setting, device)
    dp         = load_data_process(setting, device=device)
    _, test_dataset = load_datasets(setting)

    phi_steps   = setting.assumption.angle_steps_phi
    theta_steps = setting.assumption.angle_steps_theta
    max_theta   = setting.dataset.max_theta

    test_len = min(num_samples or len(test_dataset), len(test_dataset))
    print(f"Evaluating {test_len} samples ...")

    results  = []
    skipped  = 0
    filtered = 0

    for i in range(0, test_len, batch_size):
        batch_end   = min(i + batch_size, test_len)
        batch_items = [test_dataset[j] for j in range(i, batch_end)]
        csi_paths   = [x[1] for x in batch_items]

        csi_norm_list = []
        for x in batch_items:
            csi_s = x[0]
            amp   = csi_s.abs().max()
            csi_norm_list.append(csi_s / amp if amp > 0 else csi_s)
        csi_tensor = torch.stack(csi_norm_list).to(device)

        with torch.no_grad():
            z = torch.randn(
                csi_tensor.shape[0],
                setting.assumption.sample_num * setting.dataset.M * setting.dataset.N,
            ).to(device)
            raw_weights = generator(z)
            weights_out, _ = transform_weights(raw_weights)

            sample_rss_norm, scale, query_rss_pred = _run_inference(
                estimator, weights_out, csi_tensor, dp
            )
            sample_rss_norm = sample_rss_norm.cpu()
            query_rss_pred  = query_rss_pred.cpu()
            scale           = scale.cpu()

        for bidx, csi_path in enumerate(csi_paths):
            paths_df = _load_path_info(csi_path, metainfo_dir)
            if paths_df is None or len(paths_df) < NUM_PATHS:
                skipped += 1
                continue

            phi_arr   = paths_df["phi_rel"].values.astype(float)
            theta_arr = paths_df["theta_rel"].values.astype(float)

            csi_1    = csi_tensor[bidx].unsqueeze(0)
            selected = _select_top3_with_separation(csi_1, phi_arr, theta_arr, dp, device)
            if selected is None:
                skipped += 1
                continue

            # Filter: discard if 3rd-peak GT RSS < 1st-peak GT RSS / 10
            gt_rss_1 = selected[0][2]
            gt_rss_3 = selected[2][2]
            if gt_rss_3 < gt_rss_1 / 10.0:
                filtered += 1
                continue

            pred_amp_inv = arn_model(sample_rss_norm[bidx].unsqueeze(0))
            pred_lobe    = float(1.0 / torch.mean(pred_amp_inv).item())
            scale_val    = float(scale[bidx].item())
            from beamformer.utils import gpu_tensor_to_np
            pred_2d_np   = gpu_tensor_to_np(
                query_rss_pred[bidx].reshape(phi_steps, theta_steps)
            )

            entry = {"csi_path": csi_path}
            for k, (phi, theta, gt_rss_linear) in enumerate(selected):
                err = _rss_error_arn(
                    pred_2d_np, pred_lobe, scale_val,
                    phi, theta, gt_rss_linear, max_theta,
                )
                entry[f"path{k+1}_error_db"] = float(err)
                entry[f"path{k+1}_phi"]       = float(phi)
                entry[f"path{k+1}_theta"]     = float(theta)
                entry[f"path{k+1}_gt_rss_db"] = float(10.0 * np.log10(max(gt_rss_linear, 1e-30)))

            results.append(entry)

        print(
            f"  [{batch_end}/{test_len}] processed | "
            f"valid={len(results)} skipped={skipped} filtered(weak 3rd)={filtered}"
        )

    df = pd.DataFrame(results)
    print(
        f"\nDone. {len(df)} valid samples, "
        f"{skipped} skipped (missing CSV/insufficient paths), "
        f"{filtered} filtered (weak 3rd peak)."
    )
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_from_json(results_path):
    """
    Load multi-path evaluation results from JSON.

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
    Run multi-path RSS error evaluation and save results to JSON.

    Returns:
        str: path to the saved JSON file
    """
    if save_path is None:
        save_path = _DEFAULT_RESULTS_PATH

    df = _run_evaluation(num_samples=num_samples, batch_size=batch_size)

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
