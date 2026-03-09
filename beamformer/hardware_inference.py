"""
hardware_inference.py - Unified hardware inference for Sivers and IBM SDR.

Merges hardware_eval_sivers/main.py and hardware_eval_ibm/main.py into a single
coordinator that evaluates all real-world hardware scenarios.

Usage (from_scratch):
    python hardware_inference.py --device all --save_dir eval_results/realworld/<timestamp>/
    python hardware_inference.py --device sivers
    python hardware_inference.py --device ibm
"""

import os
import sys
import csv
import json
import pickle

from datetime import datetime
from typing import Optional
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import project modules
from .utils import load_config, gpu_tensor_to_np
from .modules import TransformerModel, AmplitudeRecoveryNetwork
from .dataset import load_data_process
from .weight_generator import PredefinedGenerator, ParametricGenerator


def hardware_transform_weights(B):
    """
    Hardware-specific weight transformation using phase shifting.
    Used only for hardware inference (not training).
    """
    if not torch.all((B >= -1) & (B <= 1)):
        raise ValueError("All elements in B must be in the range [-1, 1].")
    active_antenna = (B >= -1)  # all activated
    weights_output = torch.zeros_like(B, dtype=torch.complex64)
    weights_output[active_antenna] = torch.exp(torch.pi * 1j * B[active_antenna])
    return weights_output, active_antenna

# Model config names
SIVERS_CONFIG_NAME = "sivers_pi_param_co_train_16_snr_worse_indoor"
IBM_CONFIG_NAME = "ibm_param_co_train_indoor_snr_n5_32beam"

# Model paths
SIVERS_GENERATOR_PATH = "saved_models/sivers/generator_epoch_final.pth"
SIVERS_ESTIMATOR_PATH = "saved_models/sivers/estimator_epoch_final.pth"
IBM_GENERATOR_PATH = "saved_models/ibm/generator_epoch_final.pth"
IBM_ESTIMATOR_PATH = "saved_models/ibm/estimator_epoch_final.pth"

# Sivers ML RSS index slice (16 samples starting at index 8)
SIVERS_ML_POWER_IDX = np.arange(8, 24)

# Data directories
SIVERS_DATA_DIR = os.path.join(_PROJECT_ROOT, "csi-dataset/realworld_sivers")
IBM_DATA_DIR = os.path.join(_PROJECT_ROOT, "csi-dataset/realworld_ibm")

# Scenario names
SIVERS_SCENARIOS = ["indoor-los", "indoor-nlos", "outdoor-los"]
IBM_SCENARIOS = ["indoor-los", "indoor-nlos", "outdoor-los"]


# ---------------------------------------------------------------------------
# Utility functions (inlined from hardware_eval_sivers/main.py)
# ---------------------------------------------------------------------------

def watts_to_dbm(P):
    if isinstance(P, torch.Tensor):
        P = P.to(torch.float32)
        return 10 * torch.log10(P / 1e-3 + 1e-12)
    else:
        P = np.array(P, dtype=np.float32)
        return 10 * np.log10(P / 1e-3 + 1e-12)


def get_uniform_samples(N, theta_max):
    golden_ratio = (1 + np.sqrt(5)) / 2
    phi = (360 * np.arange(N) / golden_ratio) % 360
    r = np.sqrt(np.arange(N) * theta_max / 90 / N)
    theta = np.arcsin(r) * 180 / np.pi
    return [(phi[i].item(), theta[i].item()) for i in range(N)]


def shift_angle_spectrum(angle_spectrum):
    """Shift first half to back (Sivers-specific)."""
    n_phi = angle_spectrum.shape[0]
    half = n_phi // 2
    return torch.concatenate([angle_spectrum[half:], angle_spectrum[:half]], axis=0)


def uniform_indices_from_all(sweep_rss_list, baseline_num, theta_max):
    sample_points_l = get_uniform_samples(len(sweep_rss_list), theta_max)
    sample_points_s = get_uniform_samples(baseline_num, theta_max)
    points_l = np.array(sample_points_l)
    points_s = np.array(sample_points_s)
    phi_rad_l = np.radians(points_l[:, 0])
    theta_rad_l = np.radians(points_l[:, 1])
    phi_rad_s = np.radians(points_s[:, 0])
    theta_rad_s = np.radians(points_s[:, 1])
    x_l = np.sin(theta_rad_l) * np.cos(phi_rad_l)
    y_l = np.sin(theta_rad_l) * np.sin(phi_rad_l)
    z_l = np.cos(theta_rad_l)
    index_list = []
    for i in range(len(points_s)):
        x_s = np.sin(theta_rad_s[i]) * np.cos(phi_rad_s[i])
        y_s = np.sin(theta_rad_s[i]) * np.sin(phi_rad_s[i])
        z_s = np.cos(theta_rad_s[i])
        dot = np.clip(x_l * x_s + y_l * y_s + z_l * z_s, -1, 1)
        index_list.append(int(np.argmin(np.arccos(dot))))
    return index_list


def _interpolate_angle_spectrum_fn(point_coords, energy_values, theta_max, ax=None,
                                   title="", force_noplot=0):
    """Interpolate angle spectrum, matching hardware_utils.interpolate_angle_spectrum exactly."""
    N = len(point_coords)
    if len(point_coords) != len(energy_values):
        raise ValueError("point_coords and energy_values must have the same length")

    energy_array = np.array(energy_values)

    # Convert to Cartesian coordinates
    x = np.zeros((N,))
    y = np.zeros((N,))
    for i in range(N):
        r = np.sin(point_coords[i][1] * np.pi / 180)
        phi = point_coords[i][0]
        x[i] = r * np.cos(phi * np.pi / 180)
        y[i] = r * np.sin(phi * np.pi / 180)

    r_max = np.sin(theta_max * np.pi / 180)

    spacing = 0.001  # Grid spacing for interpolation
    x_range = np.arange(-r_max, r_max + spacing, spacing)
    y_range = np.arange(-r_max, r_max + spacing, spacing)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    energy_grid = np.zeros_like(x_grid)
    mask = x_grid**2 + y_grid**2 <= r_max**2

    points_card = np.column_stack((x, y))
    energy_grid_temp = griddata(
        points_card, energy_array, (x_grid, y_grid),
        method='linear', fill_value=0.0
    )

    energy_grid[mask] = energy_grid_temp[mask]
    energy_grid[~mask] = np.nan
    energy_grid = np.nan_to_num(energy_grid, nan=0.0)

    phi_grid = np.arctan2(y_grid, x_grid) * 180 / np.pi
    phi_grid[phi_grid < 0] += 360

    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    theta_grid = np.arcsin(np.clip(r_grid, 0, 1)) * 180 / np.pi

    return phi_grid, theta_grid, energy_grid, x_grid, y_grid


# ---------------------------------------------------------------------------
# IBM data loading utilities (from hardware_eval_ibm/main.py)
# ---------------------------------------------------------------------------

def _read_beam_rss_csv(csv_path):
    rss_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rss_list.append(float(row[1]))
    return rss_list


def _get_subfolder_paths(folder_path):
    return [
        os.path.join(folder_path, name) for name in sorted(os.listdir(folder_path))
        if os.path.isdir(os.path.join(folder_path, name))
    ]


def _get_csv_path_pair(parent_folder):
    first_half_folders = _get_subfolder_paths(f"{parent_folder}/combined_codebook_first_half")
    second_half_folders = _get_subfolder_paths(f"{parent_folder}/combined_codebook_second_half")
    assert len(first_half_folders) == 1 and len(second_half_folders) == 1
    return (os.path.join(first_half_folders[0], "beam_rss.csv"),
            os.path.join(second_half_folders[0], "beam_rss.csv"))


def _parse_beam_value(first_csv, second_csv):
    d1 = _read_beam_rss_csv(first_csv)
    d2 = _read_beam_rss_csv(second_csv)
    all_data = d1 + d2
    return {
        "beam_scan": [10 ** (db / 10) for db in all_data[:416]],
        "ml_32": [10 ** (db / 10) for db in all_data[416:416 + 32]],
        "ml_64": [10 ** (db / 10) for db in all_data[416 + 32:416 + 96]],
    }


def _load_ibm_experiment(experiment_path):
    """Load one IBM experiment folder and return parsed data dict."""
    first_csv, second_csv = _get_csv_path_pair(experiment_path)
    return _parse_beam_value(first_csv, second_csv)


# ---------------------------------------------------------------------------
# HardwareInferencer (unified)
# ---------------------------------------------------------------------------

class HardwareInferencer:
    """Unified hardware RSS-to-angle-spectrum inferencer for Sivers and IBM."""

    def __init__(self,
                 config_dataset: SimpleNamespace,
                 config_assumption: SimpleNamespace,
                 config_generator: SimpleNamespace,
                 config_estimator: SimpleNamespace,
                 config_arn: Optional[SimpleNamespace] = None,
                 apply_shift: bool = False,
                 device: Optional[str] = None):
        self.config_dataset = config_dataset
        self.config_assumption = config_assumption
        self.config_generator = config_generator
        self.config_estimator = config_estimator
        self.config_arn = config_arn
        self.apply_shift = apply_shift  # True for Sivers, False for IBM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.combined_config = SimpleNamespace(
            dataset=config_dataset,
            assumption=config_assumption,
            generator=config_generator,
            estimator=config_estimator,
        )
        if config_arn is not None:
            self.combined_config.arn_model = config_arn

        self._load_models()
        self._load_data_processor()

    def _load_models(self):
        self.estimator = self._init_estimator()
        self.estimator.eval()

        self.arn_model = None
        if self.config_arn is not None:
            try:
                self.arn_model = self._init_arn()
                self.arn_model.eval()
            except Exception as e:
                print(f"[HardwareInferencer] Warning: ARN load failed: {e}")

        self.generator = self._init_generator()

    def _init_estimator(self):
        estimator = TransformerModel(self.config_estimator)
        estimator.load_state_dict(
            torch.load(self.config_estimator.estimator_pretrained_model,
                       map_location=self.device, weights_only=True)
        )
        return estimator.to(self.device)

    def _init_arn(self):
        arn = AmplitudeRecoveryNetwork(
            sample_num=self.config_assumption.sample_num,
            d_model=self.config_arn.hidden_dims
        )
        if (hasattr(self.config_arn, 'ARN_model_pretrained_model') and
                self.config_arn.ARN_model_pretrained_model is not None):
            arn.load_state_dict(
                torch.load(self.config_arn.ARN_model_pretrained_model,
                           map_location=self.device, weights_only=True)
            )
        return arn.to(self.device)

    def _init_generator(self):
        sample_num = self.config_assumption.sample_num
        if self.config_generator.type == "PARAM":
            gen = ParametricGenerator(sample_num, self.config_dataset.M, self.config_dataset.N)
            if (hasattr(self.config_generator, "generator_pretrained_model") and
                    self.config_generator.generator_pretrained_model):
                gen.load_state_dict(
                    torch.load(self.config_generator.generator_pretrained_model,
                               map_location=self.device, weights_only=True)
                )
            gen.eval()
            return gen.to(self.device)
        else:
            return PredefinedGenerator(
                M_act=self.config_generator.M_act,
                N_act=self.config_generator.N_act,
                sample_mode=self.config_generator.type,
                sample_num=sample_num,
                M_base=self.config_dataset.M,
                N_base=self.config_dataset.N,
                start_freq=self.config_dataset.start_freq,
                end_freq=self.config_dataset.end_freq,
                angle_steps_theta=self.config_assumption.angle_steps_theta,
                angle_steps_phi=self.config_assumption.angle_steps_phi,
                freq_num=self.config_dataset.freq_num,
                d_row=self.config_dataset.d_row,
                d_col=self.config_dataset.d_col,
                max_theta=self.config_dataset.max_theta,
                device=self.device
            )

    def _load_data_processor(self):
        self.dp = load_data_process(self.combined_config, device=self.device)

    def _generate_weights(self, batch_size=1):
        if self.config_generator.type in ["random", "uniform", "fixed"]:
            weights, _ = self.generator.generate(batch_size=batch_size)
        else:
            z_dim = (self.config_assumption.sample_num *
                     self.config_dataset.M * self.config_dataset.N)
            z = torch.randn(batch_size, z_dim).to(self.device)
            with torch.no_grad():
                raw = self.generator(z)
                weights, _ = hardware_transform_weights(raw)
        return weights

    def infer(self, hardware_rss):
        """Run inference and return angle spectrum tensor [phi, theta]."""
        if isinstance(hardware_rss, (list, np.ndarray)):
            rss = torch.tensor(hardware_rss, dtype=torch.float32, device=self.device)
        else:
            rss = hardware_rss.to(self.device, dtype=torch.float32)
        if rss.dim() == 1:
            rss = rss.unsqueeze(0)

        weights = self._generate_weights(batch_size=rss.shape[0])
        sample_pos_enc = self.dp.generate_sample_position_encoding(weights).to(torch.float32)
        query_pos_enc = self.dp.generate_query_position_encoding(batch_size=rss.shape[0]).to(torch.float32)

        scale = torch.max(rss, dim=1, keepdim=True).values
        rss_norm = rss / scale

        with torch.no_grad():
            pred, _ = self.estimator(rss_norm, sample_pos_enc, query_pos_enc)

        if self.arn_model is not None:
            with torch.no_grad():
                pred_amp_inv = self.arn_model(rss_norm)
                lobe = 1.0 / torch.mean(pred_amp_inv).item()
        else:
            lobe = 1.0

        spectrum = pred * lobe * scale
        phi_s = self.config_assumption.angle_steps_phi
        theta_s = self.config_assumption.angle_steps_theta
        return spectrum[0].reshape(phi_s, theta_s)

    def _interpolate_spectrum(self, spectrum, target_phi=360, target_theta=90):
        sp_np = gpu_tensor_to_np(spectrum)
        phi_b, theta_b = sp_np.shape
        phi_o = np.linspace(0, 360, phi_b)
        theta_o = np.linspace(0, self.config_dataset.max_theta, theta_b)
        pm_o, tm_o = np.meshgrid(phi_o, theta_o, indexing='ij')
        phi_t = np.linspace(0, 360, target_phi)
        theta_t = np.linspace(0, self.config_dataset.max_theta, target_theta)
        pm_t, tm_t = np.meshgrid(phi_t, theta_t, indexing='ij')
        pts = np.column_stack([pm_o.ravel(), tm_o.ravel()])
        vals = sp_np.ravel()
        xi = np.column_stack([pm_t.ravel(), tm_t.ravel()])
        interp = griddata(pts, vals, xi, method='cubic', fill_value=0.0)
        return torch.tensor(interp.reshape(target_phi, target_theta),
                            device=self.device, dtype=torch.float32)

    def _spherical_dist(self, phi1, theta1, phi2, theta2):
        p1, t1, p2, t2 = map(np.radians, [phi1, theta1, phi2, theta2])
        x1, y1, z1 = np.sin(t1)*np.cos(p1), np.sin(t1)*np.sin(p1), np.cos(t1)
        x2, y2, z2 = np.sin(t2)*np.cos(p2), np.sin(t2)*np.sin(p2), np.cos(t2)
        return np.arccos(np.clip(x1*x2 + y1*y2 + z1*z2, -1, 1))

    def _find_peak(self, spectrum):
        idx = torch.argmax(spectrum)
        phi_b, theta_b = spectrum.shape
        pi, ti = np.unravel_index(idx.cpu().numpy(), (phi_b, theta_b))
        return (pi * 360.0 / phi_b,
                ti * self.config_dataset.max_theta / theta_b)

    def evaluate(self, dl_rss, sweep_rss):
        """
        Run hardware evaluation and return results dict with RSS loss and angle error.
        Returns dict with our_rss_loss, baseline_rss_loss, our_angle_error, baseline_angle_error.
        """
        theta_max = self.config_dataset.max_theta
        sweep_dirs = get_uniform_samples(len(sweep_rss), theta_max)

        raw_spectrum = self.infer(dl_rss)
        spectrum = self._interpolate_spectrum(raw_spectrum, 360, 90)
        if self.apply_shift:
            spectrum = shift_angle_spectrum(spectrum)

        our_peak_phi, our_peak_theta = self._find_peak(spectrum)

        # Find closest sweep direction to our prediction
        dists = [self._spherical_dist(our_peak_phi, our_peak_theta, d[0], d[1])
                 for d in sweep_dirs]
        our_idx = int(np.argmin(dists))
        our_rss_dbm = float(watts_to_dbm(sweep_rss[our_idx]))

        # Baseline: uniform subset
        baseline_indices = uniform_indices_from_all(sweep_rss, len(dl_rss), theta_max)
        baseline_rss_vals = [sweep_rss[i] for i in baseline_indices]
        best_bl_idx = baseline_indices[int(np.argmax(baseline_rss_vals))]
        baseline_rss_dbm = float(watts_to_dbm(sweep_rss[best_bl_idx]))
        baseline_dir = sweep_dirs[best_bl_idx]

        # Ground truth: interpolate sweep to find true peak
        phi_g, theta_g, energy_g, _, _ = _interpolate_angle_spectrum_fn(
            sweep_dirs, sweep_rss, theta_max, force_noplot=1
        )
        valid = ~np.isnan(energy_g)
        if not np.any(valid):
            raise ValueError("All NaN in interpolated energy grid")
        max_idx = np.unravel_index(np.nanargmax(energy_g), energy_g.shape)
        gt_phi = float(phi_g[max_idx])
        gt_theta = float(theta_g[max_idx])
        gt_rss_dbm = float(watts_to_dbm(energy_g[max_idx]))

        our_rss_loss = abs(gt_rss_dbm - our_rss_dbm)
        baseline_rss_loss = abs(gt_rss_dbm - baseline_rss_dbm)
        our_angle_error = float(np.degrees(self._spherical_dist(our_peak_phi, our_peak_theta, gt_phi, gt_theta)))
        baseline_angle_error = float(np.degrees(self._spherical_dist(baseline_dir[0], baseline_dir[1], gt_phi, gt_theta)))

        return {
            "our_rss_loss": our_rss_loss,
            "baseline_rss_loss": baseline_rss_loss,
            "our_angle_error": our_angle_error,
            "baseline_angle_error": baseline_angle_error,
            "gt_rss_dbm": gt_rss_dbm,
        }

    def get_spectrum_for_plot(self, dl_rss, sweep_rss):
        """Return (sweep_dirs, sweep_rss, angle_spectrum) for compare_scan_with_pred."""
        theta_max = self.config_dataset.max_theta
        sweep_dirs = get_uniform_samples(len(sweep_rss), theta_max)
        raw_spectrum = self.infer(dl_rss)
        spectrum = self._interpolate_spectrum(raw_spectrum, 360, 90)
        if self.apply_shift:
            spectrum = shift_angle_spectrum(spectrum)
        return sweep_dirs, sweep_rss, spectrum


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_sivers_inferencer(device=None):
    """Build HardwareInferencer for Sivers SDR."""
    config = load_config(SIVERS_CONFIG_NAME)
    config.generator.generator_pretrained_model = os.path.join(_PROJECT_ROOT, SIVERS_GENERATOR_PATH)
    config.estimator.estimator_pretrained_model = os.path.join(_PROJECT_ROOT, SIVERS_ESTIMATOR_PATH)
    return HardwareInferencer(
        config.dataset, config.assumption, config.generator, config.estimator,
        apply_shift=True, device=device
    )


def build_ibm_inferencer(device=None):
    """Build HardwareInferencer for IBM SDR."""
    config = load_config(IBM_CONFIG_NAME)
    config.generator.generator_pretrained_model = os.path.join(_PROJECT_ROOT, IBM_GENERATOR_PATH)
    config.estimator.estimator_pretrained_model = os.path.join(_PROJECT_ROOT, IBM_ESTIMATOR_PATH)
    return HardwareInferencer(
        config.dataset, config.assumption, config.generator, config.estimator,
        apply_shift=False, device=device
    )


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def run_sivers_evaluation(data_dir=None, inferencer=None):
    """
    Evaluate Sivers hardware data across all scenarios.
    Returns dict: scenario_name -> list of result dicts.
    """
    if data_dir is None:
        data_dir = SIVERS_DATA_DIR
    if inferencer is None:
        print("[Sivers] Building inferencer...")
        inferencer = build_sivers_inferencer()

    results = {}
    for scenario in SIVERS_SCENARIOS:
        scenario_dir = os.path.join(data_dir, scenario)
        if not os.path.isdir(scenario_dir):
            print(f"[Sivers] Skipping {scenario}: directory not found")
            continue
        pkl_files = sorted([f for f in os.listdir(scenario_dir) if f.endswith(".pkl")])
        scenario_results = []
        print(f"[Sivers] Evaluating {scenario}: {len(pkl_files)} samples")
        for pkl_file in pkl_files:
            pkl_path = os.path.join(scenario_dir, pkl_file)
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                dl_rss = list(data["ml_rss"][SIVERS_ML_POWER_IDX])
                sweep_rss = list(data["sweep_rss"])
                res = inferencer.evaluate(dl_rss, sweep_rss)
                scenario_results.append(res)
            except Exception as e:
                print(f"[Sivers] Error on {pkl_file}: {e}")
        results[scenario] = scenario_results
        print(f"[Sivers] {scenario}: {len(scenario_results)} successful evaluations")
    return results


def run_ibm_evaluation(data_dir=None, inferencer=None):
    """
    Evaluate IBM hardware data across all scenarios.
    Returns dict: scenario_name -> list of result dicts.
    """
    if data_dir is None:
        data_dir = IBM_DATA_DIR
    if inferencer is None:
        print("[IBM] Building inferencer...")
        inferencer = build_ibm_inferencer()

    results = {}
    for scenario in IBM_SCENARIOS:
        scenario_dir = os.path.join(data_dir, scenario)
        # Find all beamtable directories
        beamtable_dirs = [d for d in _get_subfolder_paths(scenario_dir)
                         if os.path.isdir(d)]
        scenario_results = []
        for bt_dir in beamtable_dirs:
            experiment_dirs = _get_subfolder_paths(bt_dir)
            for exp_dir in experiment_dirs:
                if "special" in os.path.basename(exp_dir):
                    continue  # skip experiment_special (used for Figure 20)
                try:
                    data = _load_ibm_experiment(exp_dir)
                    dl_rss = data["ml_32"]
                    sweep_rss = data["beam_scan"]
                    res = inferencer.evaluate(dl_rss, sweep_rss)
                    scenario_results.append(res)
                except Exception as e:
                    pass  # skip malformed experiments
        results[scenario] = scenario_results
        print(f"[IBM] {scenario}: {len(scenario_results)} successful evaluations")
    return results


def run_all_evaluation(save_dir):
    """Run full hardware evaluation and save results to JSON."""
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Running Sivers SDR evaluation...")
    print("=" * 60)
    sivers_inferencer = build_sivers_inferencer()
    sivers_results = run_sivers_evaluation(inferencer=sivers_inferencer)

    print("=" * 60)
    print("Running IBM SDR evaluation...")
    print("=" * 60)
    ibm_inferencer = build_ibm_inferencer()
    ibm_results = run_ibm_evaluation(inferencer=ibm_inferencer)

    all_results = {
        "sivers": sivers_results,
        "ibm": ibm_results,
    }

    json_path = os.path.join(save_dir, "hardware_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved hardware results to: {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Spectrum plot helpers (for Figure 19 and 20)
# ---------------------------------------------------------------------------
    
def compare_scan_with_pred(point_coords, energy_values, pred_tensor, 
                          theta_max=60, save_path=None, figsize=(16, 8),
                          title_prefix="", colormap='viridis',
                          show_angle_lines=False, show_peak_coordinates=False,
                          show_colorbar=False, show_colorbar_label=False):

    if len(point_coords) != len(energy_values):
        raise ValueError("point_coords and energy_values must have the same length")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
    ax1.set_facecolor('white') 
    
    N = len(point_coords)
    phi_points = np.array([coord[0] for coord in point_coords])
    theta_points = np.array([coord[1] for coord in point_coords])
    energy_array = np.array(energy_values)
    
    x = np.zeros((N,))
    y = np.zeros((N,))
    for i in range(len(point_coords)):
        r = np.sin(point_coords[i][1] * np.pi / 180)
        phi = point_coords[i][0]
        x[i] = r * np.cos(phi * np.pi / 180)
        y[i] = r * np.sin(phi * np.pi / 180)
    
    r_max = np.sin(theta_max * np.pi / 180)
    
    spacing = 0.001  
    x_range = np.arange(-r_max, r_max + spacing, spacing)
    y_range = np.arange(-r_max, r_max + spacing, spacing)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    energy_grid = np.zeros_like(x_grid)
    mask = x_grid**2 + y_grid**2 <= r_max**2
    
    points_card = np.column_stack((x, y))
    
    energy_grid_temp = griddata(
        points_card, 
        energy_array, 
        (x_grid, y_grid), 
        method='linear',
        fill_value=0.0
    )
    
    energy_grid[mask] = energy_grid_temp[mask]
    energy_grid[~mask] = np.nan 
    phi_grid = np.arctan2(y_grid, x_grid) * 180 / np.pi
    phi_grid[phi_grid < 0] += 360
    
    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    theta_grid = np.arcsin(np.clip(r_grid, 0, 1)) * 180 / np.pi
    
    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad(color='none') 
    
    im1 = ax1.imshow(energy_grid, extent=[-r_max, r_max, -r_max, r_max],
                     origin='lower', cmap=cmap, aspect='equal')
    
    ax1.scatter(x, y, c='red', s=20, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=10)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_title('Ground Truth', fontsize=18)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax1.set_xlim(-r_max, r_max)
    ax1.set_ylim(-r_max, r_max)
    ax1.set_aspect('equal', adjustable='box')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    if show_colorbar:
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        if show_colorbar_label:
            cbar1.set_label('Energy Intensity')
    

    ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    pred_np = gpu_tensor_to_np(pred_tensor)
    
    phi_bins, theta_bins = pred_np.shape
    
    phi_values = np.linspace(0, 360, phi_bins)
    theta_values = np.linspace(0, theta_max, theta_bins)
    
    phi_rad = np.linspace(0, 2*np.pi, phi_bins)
    theta_norm = np.linspace(0, 1, theta_bins)
    
    THETA, R = np.meshgrid(phi_rad, theta_norm)
    
    pred_normalized = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
    
    levels = np.linspace(0, np.max(pred_normalized), 50)
    im2 = ax2.contourf(THETA, R, pred_normalized.T, levels=levels, cmap=colormap, alpha=0.9)
    
    ax2.set_ylim(0, 1)
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(1)
    
    ax2.set_thetagrids([])
    ax2.set_rgrids([])
    ax2.grid(False)
    ax2.set_title('Prediction', fontsize=18)
    
    if show_peak_coordinates:
        max_idx = np.unravel_index(pred_normalized.argmax(), pred_normalized.shape)
        peak_phi_idx = max_idx[0]
        peak_theta_idx = max_idx[1]
        peak_phi_deg = phi_values[peak_phi_idx]
        peak_theta_deg = theta_values[peak_theta_idx]
        peak_phi_rad = np.radians(peak_phi_deg)
        peak_r = peak_theta_deg / theta_max
        
        ax2.plot(peak_phi_rad, peak_r, 'r*', markersize=12, 
                label=f'Peak: θ={peak_theta_deg:.1f}°, φ={peak_phi_deg:.1f}°')
        ax2.legend(loc='upper left', bbox_to_anchor=(0.05, 1.1), fontsize=10)
    
    if show_colorbar:
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.1)
        if show_colorbar_label:
            cbar2.set_label('Normalized Intensity')
        cbar2.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar2.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)  
    else:
        plt.show()
        plt.close(fig) 
    
    interpolated_data = {
        'phi_grid': phi_grid,
        'theta_grid': theta_grid,
        'energy_grid': energy_grid,
        'x_grid': x_grid,
        'y_grid': y_grid
    }
    
    return fig, (ax1, ax2), interpolated_data


def plot_sivers_spectrum(pkl_path, save_path=None):
    """
    Load a Sivers pkl sample and plot compare_scan_with_pred (Figure 19).
    """
    inferencer = build_sivers_inferencer()
    theta_max = inferencer.config_dataset.max_theta  # 40

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    dl_rss = list(data["ml_rss"][SIVERS_ML_POWER_IDX])
    sweep_rss = list(data["sweep_rss"])

    sweep_dirs, sweep_rss_out, spectrum = inferencer.get_spectrum_for_plot(dl_rss, sweep_rss)
    fig, (ax1, ax2), _ = compare_scan_with_pred(
        sweep_dirs, sweep_rss_out, spectrum,
        theta_max=theta_max, save_path=save_path
    )
    return fig


def plot_ibm_spectrum(experiment_path, save_path=None):
    """
    Load an IBM experiment folder and plot compare_scan_with_pred (Figure 20).
    """
    inferencer = build_ibm_inferencer()
    theta_max = inferencer.config_dataset.max_theta  # 60

    data = _load_ibm_experiment(experiment_path)
    dl_rss = data["ml_32"]
    sweep_rss = data["beam_scan"]

    sweep_dirs, sweep_rss_out, spectrum = inferencer.get_spectrum_for_plot(dl_rss, sweep_rss)
    fig, (ax1, ax2), _ = compare_scan_with_pred(
        sweep_dirs, sweep_rss_out, spectrum,
        theta_max=theta_max, save_path=save_path
    )
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run hardware inference for Sivers and/or IBM SDR."
    )
    parser.add_argument(
        "--device", choices=["sivers", "ibm", "all"], default="all",
        help="Which hardware to evaluate"
    )
    parser.add_argument(
        "--save_dir", default=None,
        help="Directory to save results JSON (default: eval_results/realworld/<timestamp>/)"
    )
    args = parser.parse_args()

    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(
            _PROJECT_ROOT, "eval_results", "realworld", timestamp
        )

    if args.device == "all":
        run_all_evaluation(args.save_dir)
    elif args.device == "sivers":
        os.makedirs(args.save_dir, exist_ok=True)
        sivers_inferencer = build_sivers_inferencer()
        sivers_results = run_sivers_evaluation(inferencer=sivers_inferencer)
        json_path = os.path.join(args.save_dir, "hardware_results.json")
        with open(json_path, "w") as f:
            json.dump({"sivers": sivers_results}, f, indent=2, default=str)
        print(f"Saved to: {json_path}")
    elif args.device == "ibm":
        os.makedirs(args.save_dir, exist_ok=True)
        ibm_inferencer = build_ibm_inferencer()
        ibm_results = run_ibm_evaluation(inferencer=ibm_inferencer)
        json_path = os.path.join(args.save_dir, "hardware_results.json")
        with open(json_path, "w") as f:
            json.dump({"ibm": ibm_results}, f, indent=2, default=str)
        print(f"Saved to: {json_path}")
