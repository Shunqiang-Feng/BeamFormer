import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as light_speed
from scipy.io import loadmat
import os
from importlib import import_module
import types
from datetime import datetime
from scipy.interpolate import griddata


class antenna_info():
    def __init__(self, start_freq, end_freq, freq_num, M, N, d_row, d_col, max_theta):
        """
            M : int - 行数量
            N : int - 列数量
            antenna_spacing_row: actual value
            antenna_spacing_col: actual value
        """
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.freq_num = freq_num
        self.M = M
        self.N = N
        self.max_theta = max_theta

        self.mid_wavelen = light_speed/(0.5*(start_freq + end_freq))
        self.freq_range = np.linspace(start_freq, end_freq, freq_num)

        self.d_row = d_row
        self.d_col = d_col

    def generate_angles(self, theta_steps, phi_steps):
        # return [M, N, phi_steps, theta_steps]
        thetas = np.linspace(0, self.max_theta, theta_steps)
        phis = np.linspace(0, 360, phi_steps, endpoint=False)
        thetas = np.radians(thetas)
        phis = np.radians(phis)

        steering_matrix = steering_vector_matrix_single_f(rows_num=self.M, cols_num=self.N,
                    d_row=self.d_row, d_col=self.d_col, wavelength=self.mid_wavelen, thetas=thetas, phis=phis)
        return steering_matrix

    def generate_toward_angles(self, thetas, phis):
        # return [M, N, phis, thetas]
        thetas = np.radians(thetas)
        phis = np.radians(phis)

        steering_matrix = steering_vector_matrix_single_f(rows_num=self.M, cols_num=self.N,
                    d_row=self.d_row, d_col=self.d_col, wavelength=self.mid_wavelen, thetas=thetas, phis=phis)
        return steering_matrix

    def act_1st_antenna_vector(self, K):
        mat2 = np.zeros((self.M*self.N, K), dtype=np.complex128)
        mat2[0, :] = 1
        return mat2

    def get_af_vector(self, weights, theta_steps, phi_steps, device):
        return calculate_antenna_factor_vector(self.M, self.N, weights, self.d_row, self.d_col, self.mid_wavelen, self.max_theta, theta_steps, phi_steps, device)


def steering_vector_matrix_single_f(rows_num, cols_num, d_row, d_col, wavelength, thetas, phis):

    k = 2 * np.pi / wavelength

    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing='ij')

    direction_vectors = np.stack([
        np.sin(theta_grid) * np.cos(phi_grid),
        np.sin(theta_grid) * np.sin(phi_grid)
    ], axis=-1)  # shape: [360, 91, 2]

    steering_matrix = np.zeros((rows_num, cols_num, len(phis), len(thetas)), dtype=complex)

    for m in range(rows_num):
        for n in range(cols_num):
            phase_delays = -(
                n * d_col * direction_vectors[..., 0] +
                m * d_row * direction_vectors[..., 1]
            )  # shape: [360, 91]

            sv = np.exp(1j * k * phase_delays)

            steering_matrix[m, n, :, :] = sv

    return steering_matrix


def calculate_antenna_factor_vector(M, N, weights, d_row, d_col, lambda_, theta_max, theta_steps, phi_steps, device):

    k = 2 * np.pi / lambda_

    if isinstance(weights, np.ndarray):
        weights = torch.tensor(weights, dtype=torch.cfloat, device=device).clone()
    else:
        weights = weights.to(device)
    weights_torch = weights.permute(2, 0, 1)  # [K, M, N]

    x = torch.arange(N, device=device) * d_col
    y = torch.arange(M, device=device) * d_row
    X, Y = torch.meshgrid(x, y, indexing='xy')

    theta = torch.linspace(0, np.pi * theta_max / 180, theta_steps, device=device)
    phi = torch.linspace(0, 2 * np.pi, phi_steps, device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')  # (theta_steps, phi_steps)

    kx = k * torch.sin(theta) * torch.cos(phi)
    ky = k * torch.sin(theta) * torch.sin(phi)

    phase_matrix = torch.exp(1j * (kx[..., None, None] * X + ky[..., None, None] * Y)).to(dtype=weights_torch.dtype)  # (theta_steps, phi_steps, M, N)

    AF = torch.einsum('tpmn,kmn->ktp', phase_matrix, weights_torch)  # (K, theta_steps, phi_steps)
    AF_magnitude = torch.abs(AF)
    AF_magnitude_max = torch.amax(AF_magnitude, dim=(1, 2), keepdim=True)
    AF_magnitude_all = AF_magnitude / AF_magnitude_max
    return AF_magnitude_all, AF_magnitude_max.cpu()


def beamform_complex_single_f_4_wg(beam_complex_rx, csi_complex, beam_complex_tx):
    intermediate = torch.einsum('bst,bfrt->bsfr', beam_complex_tx, csi_complex)
    angle_spectrum = torch.einsum('bsfr,bsr->bsf', intermediate, beam_complex_rx)
    abs_angle_spectrum = torch.abs(angle_spectrum)
    squared_angle_spectrum = abs_angle_spectrum ** 2
    mean_angle_spectrum = torch.mean(squared_angle_spectrum, dim=2)

    return mean_angle_spectrum


def read_csi_file_to_torch(file_path, frequency_num=128, M_tx=16, N_tx=16, M_rx=2, N_rx=1):

    mat = loadmat(file_path)
    if 'csi' in mat:
        csi_mat = mat['csi']
        csi_mat = torch.from_numpy(csi_mat).reshape((frequency_num, N_rx, M_rx, N_tx, M_tx)).permute(0, 2, 1, 4, 3).flatten(start_dim=1, end_dim=2).flatten(start_dim=2, end_dim=3)
    elif 'csi_data' in mat:
        # Already shaped (freq, rx, tx_flat), e.g. (128, 2, 256)
        csi_mat = torch.from_numpy(mat['csi_data'])
    else:
        raise KeyError(f"Neither 'csi' nor 'csi_data' found in {file_path}. Keys: {list(mat.keys())}")
    return csi_mat.to(dtype=torch.complex128)


def scale_in_last_dim(torch_vector):
    vec_min = torch_vector.min(dim=-1, keepdim=True).values
    vec_max = torch_vector.max(dim=-1, keepdim=True).values
    output = (torch_vector - vec_min) / (vec_max - vec_min + 1e-8)
    return output, vec_max - vec_min + 1e-8


def count_parameters(model, name):
    sum_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"**{name}** Total parameters: {sum_para/1e6} M")
    total = sum(p.data.sum() for p in model.parameters() if p.requires_grad)
    print(f"Sum of parameter values: {total.item():.6f}")


def load_config(config_name, predix=None):
    module_path = f"configs.{predix}.{config_name}" if predix else f"configs.{config_name}"
    return import_module(module_path).config


def normalize_weights(weights):
    """Normalize weights so that the sum of squared magnitudes equals 1 along the last dimension."""
    power = torch.sum(torch.abs(weights) ** 2, dim=-1, keepdim=True)
    power = torch.clamp(power, min=1e-12)
    return weights / torch.sqrt(power)


def get_uniform_samples(N, theta_max=90):
    golden_ratio = (1 + np.sqrt(5)) / 2
    phi = (360 * np.arange(N) / golden_ratio) % 360
    r = np.sqrt(np.arange(N) * theta_max / 90 / N)
    theta = np.arcsin(r) * 180 / np.pi
    sample_points = []

    for i in range(N):
        sample_points.append((phi[i].item(), theta[i].item()))

    return sample_points


def add_thermal_noise(csi, subcarrier_bw=240e3, snr_db=None, temperature=290):

    k_B = 1.380649e-23
    B = subcarrier_bw
    noise_power = k_B * temperature * B

    signal_power = torch.mean(torch.abs(csi) ** 2).item()
    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

    noise_std = (noise_power / 2) ** 0.5

    noise_real = torch.randn_like(csi.real) * noise_std
    noise_imag = torch.randn_like(csi.imag) * noise_std
    noise = torch.complex(noise_real, noise_imag)

    return csi + noise


def namespace_to_dict(ns):

    if isinstance(ns, types.SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns


def get_log_path():
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join('logs', date_str)
    os.makedirs(log_dir, exist_ok=True)

    time_str = datetime.now().strftime("%H-%M-%S")
    log_path = os.path.join(log_dir, f"{time_str}.log")
    return log_path


def gpu_tensor_to_np(x):
    return x.clone().detach().cpu().numpy()


def get_db(np_array):
    return 10 * np.log10(np_array + 1e-16)


# ---------------------------------------------------------------------------
# Polar disk plotting helpers
# ---------------------------------------------------------------------------

def _draw_angle_annotations(ax, r_max, max_theta):
    """Overlay theta circles and phi radial lines on a disk subplot.

    Theta rings at 30°, 60°, 90° (r = sin(theta)).
    Phi lines every 30° from 0° to 330°.
    """
    theta_marks = [t for t in [30, 60, 90] if t <= max_theta]
    for t_deg in theta_marks:
        r = np.sin(np.radians(t_deg))
        circle = plt.Circle((0, 0), r, color='white', fill=False,
                             linewidth=0.8, linestyle='--', alpha=0.6, zorder=5)
        ax.add_patch(circle)
        ax.text(r + 0.01, 0.01, f'{t_deg}°', color='white', fontsize=7,
                va='bottom', ha='left', zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='none', ec='none'))

    for phi_deg in range(0, 360, 30):
        phi_rad = np.radians(phi_deg)
        x_end = r_max * np.cos(phi_rad)
        y_end = r_max * np.sin(phi_rad)
        ax.plot([0, x_end], [0, y_end], color='white', linewidth=0.8,
                linestyle='--', alpha=0.6, zorder=5)
        scale = 1.07
        ax.text(r_max * scale * np.cos(phi_rad),
                r_max * scale * np.sin(phi_rad),
                f'{phi_deg}°', color='black', fontsize=7,
                va='center', ha='center', zorder=6)


def _draw_polar_disk_subplot(ax, x_pts, y_pts, values,
                             x_grid, y_grid, mask, r_max, cmap,
                             title, show_colorbar=False, show_colorbar_label=False,
                             colorbar_label='', scatter_xy=None,
                             peak_xy=None, peak_label=None,
                             show_angle_labels=False):
    """Interpolate scattered (x, y) points onto a Cartesian disk and render with imshow."""
    grid_temp = griddata(
        np.column_stack([x_pts, y_pts]),
        values,
        (x_grid, y_grid),
        method='linear',
        fill_value=0.0
    )

    grid_data = np.where(mask, grid_temp, np.nan)

    ax.set_facecolor('white')
    im = ax.imshow(grid_data, extent=[-r_max, r_max, -r_max, r_max],
                   origin='lower', cmap=cmap, aspect='equal')

    if scatter_xy is not None:
        ax.scatter(scatter_xy[0], scatter_xy[1],
                   c='red', s=20, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=10)

    if peak_xy is not None:
        ax.plot(peak_xy[0], peak_xy[1], 'r*', markersize=12, label=peak_label)
        ax.legend(loc='upper left', fontsize=10)

    ax.set_title(title, fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_angle_labels:
        _draw_angle_annotations(ax, r_max, 90)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if show_colorbar_label:
            cbar.set_label(colorbar_label)

    return im


def _tensor_to_disk_points(tensor_np, max_theta):
    """Convert a (phi_bins, theta_bins) array to (x_pts, y_pts, values) for disk plotting."""
    phi_bins, theta_bins = tensor_np.shape
    phi_values = np.linspace(0, 360, phi_bins, endpoint=False)
    theta_values = np.linspace(0, max_theta, theta_bins)
    PHI_GRID, THETA_GRID = np.meshgrid(np.radians(phi_values), np.radians(theta_values), indexing='ij')
    r = np.sin(THETA_GRID)
    x_pts = (r * np.cos(PHI_GRID)).ravel()
    y_pts = (r * np.sin(PHI_GRID)).ravel()
    normalized = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min() + 1e-8)
    return x_pts, y_pts, normalized.ravel()


def _make_disk_grid(max_theta, spacing=0.001):
    """Build the shared Cartesian meshgrid and circular mask for disk plotting."""
    r_max = np.sin(max_theta * np.pi / 180)
    x_range = np.arange(-r_max, r_max + spacing, spacing)
    y_range = np.arange(-r_max, r_max + spacing, spacing)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    mask = x_grid**2 + y_grid**2 <= r_max**2
    return x_grid, y_grid, mask, r_max
