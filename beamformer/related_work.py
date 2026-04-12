from .dataset import load_data_process, load_datasets
from .utils import  gpu_tensor_to_np, beamform_complex_single_f_4_wg, normalize_weights, get_tensor_db, get_db, namespace_to_filename, check_related_work_name
from .weight_generator import load_predefined_generator, PredefinedGenerator
import copy
import pandas as pd
from types import SimpleNamespace
import torch
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
import time
try:
    from skimage.transform import resize
except ImportError:
    import warnings
    warnings.warn("scikit-image not found; functions that use `resize` will fail. Install with: pip install scikit-image", ImportWarning)
    resize = None
from .utils import get_uniform_samples

try:
    import matlab
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False

def find_closest_direction_index(target_phi, target_theta, sample_points):
    """
    Find the index of the closest direction to the target direction.
    
    Args:
        target_phi: Target azimuth angle in degrees
        target_theta: Target elevation angle in degrees
        sample_points: List of tuples [(phi, theta), ...] in degrees
    
    Returns:
        Index of the closest direction
    """
    target_phi_rad = np.radians(target_phi)
    target_theta_rad = np.radians(target_theta)
    
    # Convert target to Cartesian coordinates
    x_target = np.sin(target_theta_rad) * np.cos(target_phi_rad)
    y_target = np.sin(target_theta_rad) * np.sin(target_phi_rad)
    z_target = np.cos(target_theta_rad)
    
    min_distance = float('inf')
    closest_idx = 0
    
    for idx, (phi, theta) in enumerate(sample_points):
        phi_rad = np.radians(phi)
        theta_rad = np.radians(theta)
        
        # Convert to Cartesian coordinates
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        # Calculate dot product and spherical distance
        dot_product = np.clip(x*x_target + y*y_target + z*z_target, -1.0, 1.0)
        distance = np.arccos(dot_product)
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = idx
    
    return closest_idx

def calculate_angle_error(gt_phi, gt_theta, pred_phi, pred_theta, return_degrees=True):
    """
    Calculate spherical distance (angle error) between ground truth and predicted angles.
    
    Args:
        gt_phi: Ground truth azimuth angles (phi) in degrees - pandas Series or array-like
        gt_theta: Ground truth elevation angles (theta) in degrees - pandas Series or array-like  
        pred_phi: Predicted azimuth angles (phi) in degrees - pandas Series or array-like
        pred_theta: Predicted elevation angles (theta) in degrees - pandas Series or array-like
        return_degrees: If True, return results in degrees; if False, return in radians
        
    Returns:
        pandas Series or numpy array of spherical distances (angle errors)
    """
    # Convert to numpy arrays for vectorized operations
    phi1 = np.array(gt_phi)
    theta1 = np.array(gt_theta) 
    phi2 = np.array(pred_phi)
    theta2 = np.array(pred_theta)

    if not (np.all(theta1 < 91) and np.all(theta2 < 91)):
        raise ValueError("All theta values must be less than 91 degrees.")
        
    # Convert to radians
    phi1_rad = np.radians(phi1)
    theta1_rad = np.radians(theta1)
    phi2_rad = np.radians(phi2)
    theta2_rad = np.radians(theta2)
    
    # Convert spherical to Cartesian coordinates
    # For ground truth points
    x1 = np.sin(theta1_rad) * np.cos(phi1_rad)
    y1 = np.sin(theta1_rad) * np.sin(phi1_rad)
    z1 = np.cos(theta1_rad)
    
    # For predicted points
    x2 = np.sin(theta2_rad) * np.cos(phi2_rad)
    y2 = np.sin(theta2_rad) * np.sin(phi2_rad)
    z2 = np.cos(theta2_rad)
    
    # Calculate dot product
    dot_product = x1*x2 + y1*y2 + z1*z2
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Handle numerical errors
    
    # Calculate spherical distance in radians
    spherical_distance_rad = np.arccos(dot_product)
    
    if return_degrees:
        spherical_distance = np.degrees(spherical_distance_rad)
    else:
        spherical_distance = spherical_distance_rad
    
    # Return as pandas Series if input was pandas Series
    if isinstance(gt_phi, pd.Series):
        return pd.Series(spherical_distance, index=gt_phi.index)
    else:
        return spherical_distance


def modinv(sigma, N):
    t, new_t = 0, 1
    r, new_r = N, sigma
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError(f"{sigma} - {N} Error: No inverse")
    if t < 0:
        t += N
    return t

def generate_p_pair(N):
    a = np.random.randint(0, N)
    b = np.random.randint(0, N)
    sigma = np.random.choice(np.arange(1, N, 2)).item()
    omega = np.exp(2j * np.pi / N)
    sigma_inv = modinv(sigma, N)

    P_prime = np.zeros((N, N), dtype=complex)
    for i in range(N):
        val = omega**(a * sigma * i)
        row_idx = (sigma * (i - b)) % N
        P_prime[row_idx, i] = val

    P = np.zeros((N, N), dtype=complex)
    for i in range(N):
        rho_i = (sigma_inv * i + a) % N
        P[rho_i, i] = omega ** (i*b + sigma * b * a)
    
    return P, P_prime

def get_eff_p(P):

    p = np.abs(P)

    return np.argmax(P, axis=0)

def run_agile_link(angle_spectrum, M, N, sample_num, R=2, gt_phi=None, gt_theta=None):
    # angle_spectrum phi*theta, 80*20

    L = sample_num // (M*N//(R**4))
    # print(f" parameters: M = {M}, N = {N}, sample_num = {sample_num}, L = {L}")
    def _run_agile_link_2d(x, R=2, L=4):
        """
        Internal helper function that executes the core 2D Agile-Link algorithm.
        """
        M, N = x.shape
        B_rows = M // (R**2)
        B_cols = N // (R**2)
        
        if M % (R**2) != 0 or N % (R**2) != 0:
            raise ValueError(f"Array dimensions {M}x{N} cannot be evenly divided by R^2={R**2}")

        # Pre-compute Fourier and Inverse Fourier matrices
        F_M = np.fft.fft(np.eye(M))
        F_prime_M = np.conj(F_M)
        F_N = np.fft.fft(np.eye(N))
        F_prime_N = np.conj(F_N)

        def create_initial_weights(dim, B, R):
            A_initial = np.zeros((B, dim), dtype=complex)
            P_spacing = dim // R
            for b in range(B):
                a_vec = np.zeros(dim, dtype=complex)
                for r in range(R):
                    direction = (R * b + r * P_spacing) % dim
                    start, end = r * P_spacing, (r + 1) * P_spacing
                    F_dim = np.fft.fft(np.eye(dim))
                    a_vec[start:end] = F_dim[direction, start:end]
                A_initial[b, :] = a_vec
            return A_initial

        A_rows_initial = create_initial_weights(M, B_rows, R)
        A_cols_initial = create_initial_weights(N, B_cols, R)

        final_voting_table = np.ones((M, N))

        for l in range(L):
            A_rows_current = A_rows_initial
            A_cols_current = A_cols_initial
            if l > 0:
                p_rows = np.random.permutation(M)
                p_cols = np.random.permutation(N)
                A_rows_current = A_rows_initial[:, p_rows]
                A_cols_current = A_cols_initial[:, p_cols]

            measured_powers = np.abs(A_rows_current @ F_prime_M @ x @ F_prime_N.T @ A_cols_current.T)
            measured_powers_squared = measured_powers**2
            coverage_rows = np.abs(A_rows_current @ F_prime_M)**2
            coverage_cols = np.abs(A_cols_current @ F_prime_N)**2
            
            # Matrix operation to calculate scores for this hash
            round_scores = coverage_rows.T @ measured_powers_squared @ coverage_cols
            round_scores[round_scores == 0] = 1e-9
            
            # Soft Voting (Accumulate via multiplication)
            final_voting_table *= round_scores
            
        return final_voting_table
    
    def prepare_angle_spectrum(angle_spectrum):
        # Ensure it's on CPU first
        if isinstance(angle_spectrum, torch.Tensor):
            if angle_spectrum.is_cuda:
                angle_spectrum = angle_spectrum.cpu()
            angle_spectrum = angle_spectrum.numpy()

        # Convert to numpy and transpose dims
        angle_spectrum = angle_spectrum.transpose(1, 0)  # [theta, phi]
        return angle_spectrum 


    # 1. Interpolate the low-resolution spectrum to a high-resolution 360x90 grid.
    high_res_spectrum = resize(
        prepare_angle_spectrum(angle_spectrum), # [phi, theta] -> [theta, phi]
        (90, 360), # Target shape: 90 rows for theta, 360 columns for phi
        mode='reflect',
        anti_aliasing=True
    )

    # 2. Map from the M x N grid (representing u-v space) to the signal matrix x.
    x = np.zeros((M, N))
    for m_idx in range(M):
        for n_idx in range(N):
            # Convert grid index (m, n) to direction cosines (u, v)
            v = -1 + (m_idx + 0.5) * (2 / M) # v corresponds to Y-axis / M rows
            u = -1 + (n_idx + 0.5) * (2 / N) # u corresponds to X-axis / N columns

            # Check if the (u, v) coordinate is inside the unit circle (physical limit)
            if u**2 + v**2 >= 1:
                x[m_idx, n_idx] = 0
                continue

            theta_rad = np.arcsin(np.sqrt(u**2 + v**2))
            phi_rad = np.arctan2(v, u)

            theta_deg = np.rad2deg(theta_rad)
            phi_deg = np.rad2deg(phi_rad)
            if phi_deg < 0:
                phi_deg += 360
            
            theta_lookup_idx = min(int(theta_deg), 89)
            phi_lookup_idx = min(int(phi_deg), 359)
            
            amplitude = np.sqrt(high_res_spectrum[theta_lookup_idx, phi_lookup_idx])
            x[m_idx, n_idx] = amplitude

    # 3. Run the core Agile-Link algorithm to get the voting matrix
    final_scores = _run_agile_link_2d(x, R, L)

    # 4. Find the peak score (Best Prediction)
    peak_flat_idx = np.argmax(final_scores)
    peak_m, peak_n = np.unravel_index(peak_flat_idx, (M, N))

    # Helper to convert grid index back to angles
    def grid_index_to_angles(m, n, M, N):
        v = -1 + (m + 0.5) * (2 / M)
        u = -1 + (n + 0.5) * (2 / N)
        
        if u**2 + v**2 >= 1:
            return 0.0, 0.0 

        theta_rad = np.arcsin(np.sqrt(u**2 + v**2))
        phi_rad = np.arctan2(v, u)
        
        theta_deg = np.rad2deg(theta_rad)
        phi_deg = np.rad2deg(phi_rad)
        if phi_deg < 0:
            phi_deg += 360
        return theta_deg, phi_deg

    theta_peak_deg, phi_peak_deg = grid_index_to_angles(peak_m, peak_n, M, N)

    # 5. Find the score at the grid closest to GT (if provided)
    rss_pred_at_gt = None
    if gt_phi is not None and gt_theta is not None:
        gt_phi_rad = np.deg2rad(gt_phi)
        gt_theta_rad = np.deg2rad(gt_theta)
        
        # Map spherical (theta, phi) -> direction cosines (u, v)
        # Matches the definition in Step 2: u ~ cos(phi), v ~ sin(phi)
        u_gt = np.sin(gt_theta_rad) * np.cos(gt_phi_rad)
        v_gt = np.sin(gt_theta_rad) * np.sin(gt_phi_rad)
        
        # Map (u, v) -> Grid Index (m, n)
        # Inverse of v = -1 + (m + 0.5) * (2 / M)
        m_gt_idx = int(np.round((v_gt + 1) * M / 2 - 0.5))
        # Inverse of u = -1 + (n + 0.5) * (2 / N)
        n_gt_idx = int(np.round((u_gt + 1) * N / 2 - 0.5))
        
        # Check bounds and extract the Matrix-Calculated Score
        if 0 <= m_gt_idx < M and 0 <= n_gt_idx < N:
            rss_pred_at_gt = final_scores[m_gt_idx, n_gt_idx]
        else:
            # GT is outside the FOV or physical limits captured by the grid
            rss_pred_at_gt = 0.0

    return (theta_peak_deg, phi_peak_deg), rss_pred_at_gt

def split_theta_phi(beam_num):
    if beam_num in [32, 64]:
        num_theta = 4
    elif beam_num in [8, 16]:
        num_theta = 2
    num_phi = beam_num // num_theta

    return num_phi, num_theta

def get_coarse_coverage(coarse_beam_num, max_theta):
    """
    Generate a coarse grid of beam directions (phi, theta)
    based on total beam number and angular range.

    **Note**: This function is sensitive to the input `coarse_beam_num`. 32 is a good choice for coarse beam number,
    """
    # Assume phi in [0, 360), theta in [0, max_theta]

    num_phi, num_theta = split_theta_phi(coarse_beam_num)

    phi_res = 360 / num_phi
    theta_res = max_theta / num_theta

    target_direction_coarse = []
    for i in range(num_phi):
        for j in range(num_theta):
            phi = i * phi_res + phi_res / 2
            theta = j * theta_res + theta_res / 2
            target_direction_coarse.append((phi, theta))

    if len(target_direction_coarse) != coarse_beam_num:
        raise ValueError(f"Generated {len(target_direction_coarse)} coarse directions, expected {coarse_beam_num}")

    return (phi_res, theta_res), target_direction_coarse

def determine_coarse_array_size(coarse_beam_res):
    return 4,4

def generate_fine_beam(fine_beam_num, coarse_beam_res, best_coarse_direction):
    """
    Generate fine scan directions around the best coarse direction
    with uniform sampling within the coarse sector size.
    """
    num_phi, num_theta = split_theta_phi(fine_beam_num)

    phi_center, theta_center = best_coarse_direction
    phi_res, theta_res = coarse_beam_res

    phi_start = phi_center - phi_res / 2
    theta_start = theta_center - theta_res / 2

    dphi = phi_res / num_phi
    dtheta = theta_res / num_theta

    target_direction_fine = []
    for i in range(num_phi):
        for j in range(num_theta):
            phi = phi_start + (i + 0.5) * dphi
            theta = theta_start + (j + 0.5) * dtheta
            target_direction_fine.append((phi % 360, min(max(theta, 0), 90)))  # clamp to [0, 90]

    return target_direction_fine



def interpolate_to_360_90(matrix):
 
    if matrix.shape != (80, 20):
        raise ValueError(" Input matrix must be of shape (80, 20) representing (phi, theta) grid.")

    # 原始坐标范围
    phi_orig = np.linspace(0, 360, 80, endpoint=False)  
    theta_orig = np.linspace(0, 90, 20)  

    # 构建插值器
    interpolator = RegularGridInterpolator((phi_orig, theta_orig), matrix, bounds_error=False, fill_value=None)

    # 新的插值坐标
    phi_new = np.linspace(0, 360, 360, endpoint=False)
    theta_new = np.linspace(0, 90, 90)

    # 构建插值点网格
    phi_grid, theta_grid = np.meshgrid(phi_new, theta_new, indexing='ij')
    points = np.stack([phi_grid.ravel(), theta_grid.ravel()], axis=-1)

    # 插值
    interpolated = interpolator(points).reshape(360, 90)
    return interpolated

def to_matlab_double_2d(tensor):

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if not isinstance(tensor, np.ndarray):
        raise TypeError("Input must be a 2D torch.Tensor or numpy.ndarray.")
    if tensor.ndim != 2:
        raise ValueError(f"Input must be 2D, but find {tensor.ndim}")

    return matlab.double(tensor.tolist())

def to_matlab_complex_2d(tensor, matlab_engine):


    real_part = to_matlab_double_2d(np.real(tensor))
    imag_part = to_matlab_double_2d(np.imag(tensor))

    return matlab_engine.complex(real_part, imag_part)

def from_matlab_double(mat_arr):

    np_arr = np.array(mat_arr)

    return np_arr

def beamform_2ace(query_weights, csi_single_freq):
    # query_weights: [256, 90*360]
    # csi_single_freq: [256,]
    return np.abs(np.einsum('as,a->s', query_weights, csi_single_freq))


class baseline_Hierarchical():
    def __init__(self, config_dataset, config_assumption, device):
        # For the array, M is the number of rows, N is the number of columns
        coarse_beam_num = config_assumption.sample_num // 2
        fine_beam_num = config_assumption.sample_num // 2
        self.config = SimpleNamespace(
            dataset=copy.deepcopy(config_dataset),
            assumption=copy.deepcopy(config_assumption))
        _, test_dataset = load_datasets(self.config)
        max_rss_cache = cache_max_rss(config_dataset)
        self.max_rss_performance = max_rss_cache.performance
        self.test_dataset = test_dataset
        self.dataset_folder = test_dataset.dataset_path

        self.fine_beam_num = fine_beam_num
        self.coarse_beam_res, self.target_direction_coarse = get_coarse_coverage(coarse_beam_num, self.config.dataset.max_theta)
        self.M_act_coarse, self.N_act_coarse = determine_coarse_array_size(self.coarse_beam_res)
        self.coarse_generator = self.load_predefined_generator(self.config, coarse_beam_num, self.M_act_coarse, self.N_act_coarse, device)
        self.fine_generator = self.load_predefined_generator(self.config, fine_beam_num, self.config.dataset.M, self.config.dataset.N, device)
        self.coarse_dp = self._load_data_process(self.config, device, sample_num=coarse_beam_num)
        self.fine_dp = self._load_data_process(self.config, device, sample_num=fine_beam_num)

        result_cache_folder = os.path.join('cache', 'Hierarchical', 'results')
        os.makedirs(result_cache_folder, exist_ok = True)
        self.result_path = os.path.join(result_cache_folder, namespace_to_filename(config_dataset, config_assumption, suffix='.csv'))
        if os.path.exists(self.result_path) == False:
            print(f"Hierarchical Result: Result file not found at {self.result_path}, start running to collect...")
            self.run()

            # self.performance = pd.read_csv(self.result_path) # I need to merge the max_rss_performance with the hierarchical_performance
            self.performance = pd.merge(self.max_rss_performance, self.run_result, on="csi_path")
            # add one more column for "loss"
            self.performance["loss"] = self.performance["max_rss"] - self.performance["rss"]
            self.performance['angle_error'] = calculate_angle_error(self.performance['gt_phi'], self.performance['gt_theta'], self.performance['hierarchical_direction_phi'], self.performance['hierarchical_direction_theta'])
            self.performance.to_csv(self.result_path, index=False)
        self.performance = pd.read_csv(self.result_path)
        print(f"Hierarchical Result is collected, and save to {self.result_path}")
        print("Hierarchical Result is loaded")

    def load_predefined_generator(self, config, sample_num, M_act, N_act, device):
        ds = config.dataset

        return PredefinedGenerator(
            M_act = M_act,N_act = N_act,
            sample_mode = None, sample_num = sample_num,
            M_base = ds.M, N_base = ds.N,
            start_freq = ds.start_freq, end_freq = ds.end_freq,
            angle_steps_theta = config.assumption.angle_steps_theta, angle_steps_phi = config.assumption.angle_steps_phi,
            freq_num= ds.freq_num,
            d_col=ds.d_col, d_row=ds.d_row,
            max_theta = ds.max_theta,
            device=device,
        )
    
    def _load_data_process(self, config, device, sample_num):
        config_copy = copy.deepcopy(config)
        config_copy.assumption.sample_num = sample_num
        return load_data_process(config_copy, device)

    def sector_level_sweep(self, ground_truth_csi):
        # return best coarse beam direction 
        coarse_beam_weights = self.coarse_generator.generate_weights_toward_angles(self.target_direction_coarse) # [1, sample_num, M_base, N_base] 

        rss_coarse = self.coarse_dp.generate_sample_rss(ground_truth_csi, coarse_beam_weights).squeeze(0) # [sample_num]

        # debug
        best_coarse_index = rss_coarse.argmax(dim=0)

        # af = self.coarse_dp.generate_sample_position_encoding(coarse_beam_weights)
        # plot_antenna_factor(af[0,best_coarse_index,:].reshape(16,64))


        best_coarse_direction = self.target_direction_coarse[best_coarse_index]
        return best_coarse_direction

    def fine_level_sweep(self, ground_truth_csi, best_coarse_direction):
        target_direction_fine = generate_fine_beam(self.fine_beam_num, self.coarse_beam_res, best_coarse_direction)
        fine_beam_weights = self.fine_generator.generate_weights_toward_angles(target_direction_fine) # [1, sample_num, M_base, N_base]
        rss_fine = self.fine_dp.generate_sample_rss(ground_truth_csi, fine_beam_weights).squeeze(0) # [sample_num]

        # return best fine beam rss and direction
        best_fine_index = rss_fine.argmax(dim=0)
        best_fine_direction = target_direction_fine[best_fine_index]
        best_fine_rss = rss_fine[best_fine_index]
        return best_fine_rss, best_fine_direction

    def generate(self, ground_truth_csi):
        best_coarse_direction = self.sector_level_sweep(ground_truth_csi)
        best_fine_rss, best_fine_direction = self.fine_level_sweep(ground_truth_csi, best_coarse_direction)
        return best_fine_rss, best_fine_direction
        
    def predict_rss_at_gt(self, ground_truth_csi, gt_phi, gt_theta):
        """
        Predict RSS at ground truth direction.
        First search in coarse beams, then refine if GT is in the best coarse sector.
        """
        # Level 1: Coarse beam sweep
        coarse_beam_weights = self.coarse_generator.generate_weights_toward_angles(self.target_direction_coarse)
        rss_coarse = self.coarse_dp.generate_sample_rss(ground_truth_csi, coarse_beam_weights).squeeze(0)
        
        # Find closest coarse beam to GT
        coarse_idx = find_closest_direction_index(gt_phi, gt_theta, self.target_direction_coarse)
        best_coarse_direction = self.target_direction_coarse[coarse_idx]
        
        # Scale coarse beam RSS by array size ratio
        array_size_ratio = (self.config.dataset.M * self.config.dataset.N) / (self.M_act_coarse * self.N_act_coarse)
        rss_coarse_scaled = rss_coarse[coarse_idx] * array_size_ratio
        
        # Level 2: Check if this coarse sector would be selected for refinement
        # In hierarchical search, the best coarse beam is refined
        best_coarse_idx = rss_coarse.argmax(dim=0).item()
        
        # If GT falls in the sector that would be refined, use fine beam measurement
        if coarse_idx == best_coarse_idx:
            target_direction_fine = generate_fine_beam(self.fine_beam_num, self.coarse_beam_res, best_coarse_direction)
            fine_beam_weights = self.fine_generator.generate_weights_toward_angles(target_direction_fine)
            rss_fine = self.fine_dp.generate_sample_rss(ground_truth_csi, fine_beam_weights).squeeze(0)
            
            # Find closest fine beam to GT
            fine_idx = find_closest_direction_index(gt_phi, gt_theta, target_direction_fine)
            return get_db(rss_fine[fine_idx].item())
        else:
            # Use scaled coarse beam RSS
            return get_db(rss_coarse_scaled.item())
        
    def run(self):
        performance_list = []
        for idx, (csi, csi_file_path) in enumerate(self.test_dataset):
            csi = csi.unsqueeze(0).to(self.coarse_dp.device)  # [1, M, N, F]
            best_fine_rss, best_fine_direction = self.generate(csi)
            
            # Get GT direction and predict RSS at GT
            gt_row = self.max_rss_performance[self.max_rss_performance['csi_path'] == csi_file_path]
            if not gt_row.empty:
                gt_phi = gt_row['gt_phi'].values[0]
                gt_theta = gt_row['gt_theta'].values[0]
                rss_at_gt = self.predict_rss_at_gt(csi, gt_phi, gt_theta)
            else:
                raise ValueError(f"GT direction not found for {csi_file_path}")
            
            performance_list.append({
                "csi_path": csi_file_path,
                "rss": get_db(best_fine_rss.item()),
                "hierarchical_direction_phi": best_fine_direction[0],
                "hierarchical_direction_theta": best_fine_direction[1],
                "rss_at_gt": rss_at_gt
            })
            if idx % max(1, len(self.test_dataset) // 100) == 0:
                print(f"Hierarchical Result: Processed {idx*100/len(self.test_dataset)}%  samples ...")
        self.run_result = pd.DataFrame(performance_list)
    
# below is the 2ACE baseline, which is a matlab-based baseline
class baseline_2ACE():
    def __init__(self, config_dataset, config_assumption, device):




        self.config = SimpleNamespace(
            dataset=copy.deepcopy(config_dataset),
            assumption=copy.deepcopy(config_assumption),
            generator=SimpleNamespace(
                type='random',
                M_act=config_dataset.M,
                N_act=config_dataset.N,
            )
        )
        # if sample_num != None:
        #     self.config.assumption.sample_num = sample_num
        max_rss_cache = cache_max_rss(config_dataset)
        self.max_rss_performance = max_rss_cache.performance

        _, test_dataset = load_datasets(self.config)
        self.test_dataset = test_dataset
        self.dataset_folder = test_dataset.dataset_path
        self.device = device
        if config_dataset.mode == 'rx_act1':
            self.tx_num, self.rx_num = config_dataset.M * config_dataset.N, 1
        elif config_dataset.mode == 'tx_act1':
            self.tx_num, self.rx_num = 1, config_dataset.M * config_dataset.N
        else:
            raise ValueError(f"mode {config_dataset.mode} is not supported")

        # self.config.assumption.sample_num = sample_num
        self.generator = load_predefined_generator(self.config, device=device)
        self.dp = load_data_process(self.config, device)

        result_cache_folder = os.path.join('cache', '2ACE', 'results')
        os.makedirs(result_cache_folder, exist_ok = True)
        self.result_path = os.path.join(result_cache_folder, namespace_to_filename(config_dataset, config_assumption, suffix='.csv'))
        if os.path.exists(self.result_path) == False:
            if not HAS_MATLAB:
                raise ImportError("Running 2ACE baseline requires MATLAB Engine for Python.")
            print(f"2ACE Result: Result file not found at {self.result_path}, start running to collect...")
            self.run()
            self.performance = pd.merge(self.max_rss_performance, self.run_res, on="csi_path") 
            self.performance["loss"] = self.performance["max_rss"] - self.performance["rss"]
            self.performance['angle_error'] = calculate_angle_error(self.performance['gt_phi'], self.performance['gt_theta'], self.performance['2ace_phi'], self.performance['2ace_theta'])
            self.performance.to_csv(self.result_path, index=False)
        self.performance = pd.read_csv(self.result_path)
        print(f"2ACE Result is collected, and save to {self.result_path}")
        print("2ACE Result is loaded")
        
               
    
    def run(self):
        performance_list = []
        matlab_engine = matlab.engine.start_matlab()
        matlab_engine.addpath('beamformer/2ACE_baseline')
        for idx, (csi, csi_file_path) in enumerate(self.test_dataset):
            csi = csi.unsqueeze(0).to(self.device)
            weights,_ = self.generator.generate() # [batch_size, sample_num, M_base, N_base]
            # weights = weights.view(self.config.assumption.sample_num, self.tx_num * self.rx_num)
            rss = self.dp.generate_sample_rss(csi, weights).view(-1,1)
      
            start_time = time.time()
            recover_csi_raw, _, quality= matlab_engine.inferLowRankV4(to_matlab_complex_2d(weights.view(self.config.assumption.sample_num, self.tx_num * self.rx_num), matlab_engine),
                                                                     to_matlab_double_2d(rss), float(self.tx_num), float(self.rx_num), nargout = 3)
            running_time = time.time() - start_time
            
            recover_csi = from_matlab_double(recover_csi_raw).reshape(max(self.tx_num, self.rx_num),)

            # query_weights = self.generator._generate_query_weights().reshape(max(self.tx_num, self.rx_num), -1) # [tx (or rx), 1600]
            query_weights = self.dp.antenna_info.generate_angles(self.dp.antenna_info.max_theta, 360)
            query_weights = torch.from_numpy(query_weights).reshape(self.dp.M, self.dp.N, self.dp.antenna_info.max_theta*360)
            query_weights = query_weights.unsqueeze(0).to(self.device).reshape(max(self.tx_num, self.rx_num), -1)
            
            query_weights = gpu_tensor_to_np(query_weights)
            full_as = beamform_2ace(query_weights, recover_csi).reshape(360, self.dp.antenna_info.max_theta) # [90*360,]
            # visual_2d_tensor(full_as)
            phi_opt, theta_opt = np.unravel_index(np.argmax(full_as), full_as.shape)
            # print(f"I select phi = {phi_opt}, theta = {theta_opt}")
            weight_opt = self.generator.antenna_info.generate_toward_angles(thetas=[theta_opt], phis=[phi_opt]) # M, N, 1, 1 
            weight_opt = torch.from_numpy(weight_opt).to("cuda").view(1,1,-1)
            weight_opt = normalize_weights(weight_opt)
            act1_weights = self.dp.antenna_info_act1.act_1st_antenna_vector(1)
            act1_weights = torch.from_numpy(act1_weights).to("cuda").reshape(1,1,-1)
            if self.tx_num > self.rx_num:
                rss = beamform_complex_single_f_4_wg(act1_weights, csi, weight_opt)
            else:
                rss = beamform_complex_single_f_4_wg(weight_opt, csi, act1_weights)
            rss = get_tensor_db(rss)

            # Predict RSS at ground truth direction
            gt_row = self.max_rss_performance[self.max_rss_performance['csi_path'] == csi_file_path]
            if not gt_row.empty:
                gt_phi = gt_row['gt_phi'].values[0]
                gt_theta = gt_row['gt_theta'].values[0]
                
                # Generate steering vector toward GT direction
                weight_gt = self.generator.antenna_info.generate_toward_angles(
                    thetas=[gt_theta], phis=[gt_phi]
                )  # M, N, 1, 1
                weight_gt_np = weight_gt.reshape(-1)  # Flatten to [tx or rx]
                
                # Beamform using recovered CSI in GT direction
                rss_at_gt_complex = np.dot(weight_gt_np, recover_csi)
                rss_at_gt = get_db(np.abs(rss_at_gt_complex))
            else:
                raise ValueError(f"GT direction not found for {csi_file_path}")

            # print(f"the predict rss: {rss}, the gt max rss: {10*np.log10(gpu_tensor_to_np(self.dp.generate_max_rss(csi)))}")
            performance_list.append({
                "csi_path": csi_file_path,
                "rss": rss.item(),
                "2ace_quality": quality,
                "2ace_running_time": running_time,
                "2ace_phi": float(phi_opt),
                "2ace_theta": float(theta_opt),
                "rss_at_gt": rss_at_gt
            })
            # print every 1%
            if idx % max(1, len(self.test_dataset) // 100) == 0:
                print(f"2ACE Result: Processed {idx*100/len(self.test_dataset)}%  samples ...")
        matlab_engine.quit()
        self.run_res = pd.DataFrame(performance_list)
        # df.to_csv(self.result_path, index=False)

# 802ad search baseline, in fact, this is **hierarchical**
class baseline_802ad():
    def __init__(self, config_dataset, config_assumption, device):
        self.config = SimpleNamespace(
            dataset=copy.deepcopy(config_dataset),
            assumption=copy.deepcopy(config_assumption),
            generator=SimpleNamespace(
                type='uniform',
                M_act=config_dataset.M,
                N_act=config_dataset.N
            )
        )
        max_rss_cache = cache_max_rss(config_dataset)
        self.max_rss_performance = max_rss_cache.performance
        _, test_dataset = load_datasets(self.config)
        self.test_dataset = test_dataset
        self.dataset_folder = test_dataset.dataset_path
        self.device = device
        self.generator = load_predefined_generator(self.config, device=device)
        self.dp = load_data_process(self.config, device)
        result_cache_folder = os.path.join('cache', '802ad', 'results')
        os.makedirs(result_cache_folder, exist_ok = True)
        self.result_path = os.path.join(result_cache_folder, namespace_to_filename(config_dataset, config_assumption, suffix='.csv'))
        if os.path.exists(self.result_path) == False:
            print(f"802ad Result: Result file not found at {self.result_path}, start running to collect...")
            self.run()
            self.performance = pd.merge(self.max_rss_performance, self.run_res, on="csi_path") 
            self.performance["loss"] = self.performance["max_rss"] - self.performance["rss"]
            self.performance['angle_error'] = calculate_angle_error(self.performance['gt_phi'], self.performance['gt_theta'], self.performance['ad_phi'], self.performance['ad_theta'])
            self.performance.to_csv(self.result_path, index=False)
        self.performance = pd.read_csv(self.result_path)
        print(f"802ad Result is collected, and save to {self.result_path}")
        print("802ad Result is loaded")

    def generate(self, csi):
        weights, _ = self.generator.generate()
        rss = self.dp.generate_sample_rss(csi, weights)

        rss_max_idx = np.argmax(gpu_tensor_to_np(rss.squeeze(0)))
        sample_points = get_uniform_samples(self.generator.sample_num, self.generator.antenna_info.max_theta)
        ad_phi, ad_theta = sample_points[rss_max_idx]
        
        return np.max(get_tensor_db(rss)), ad_phi, ad_theta
    
    def predict_rss_at_gt(self, csi, gt_phi, gt_theta):
        """
        Predict RSS at ground truth direction.
        For exhaustive search, find the closest measured direction.
        """
        weights, _ = self.generator.generate()
        rss = self.dp.generate_sample_rss(csi, weights)
        sample_points = get_uniform_samples(self.generator.sample_num, self.generator.antenna_info.max_theta)
        
        # Find closest measured direction to GT
        closest_idx = find_closest_direction_index(gt_phi, gt_theta, sample_points)
        rss_at_gt = rss.squeeze(0)[closest_idx]
        
        return get_db(rss_at_gt.item())
        
    def run(self):
        performance_list = []
        for idx, (csi, csi_file_path) in enumerate(self.test_dataset):
            csi = csi.unsqueeze(0).to(self.device)  # [1, M, N, F]
            rss, ad_phi, ad_theta = self.generate(csi)
            
            # Get GT direction and predict RSS at GT
            gt_row = self.max_rss_performance[self.max_rss_performance['csi_path'] == csi_file_path]
            if not gt_row.empty:
                gt_phi = gt_row['gt_phi'].values[0]
                gt_theta = gt_row['gt_theta'].values[0]
                rss_at_gt = self.predict_rss_at_gt(csi, gt_phi, gt_theta)
            else:
                raise ValueError(f"GT direction not found for {csi_file_path}")
            
            performance_list.append({
                "csi_path": csi_file_path,
                "rss": rss.item(),
                "ad_phi": ad_phi,
                "ad_theta": ad_theta,
                "rss_at_gt": rss_at_gt
            })
            if idx % max(1, len(self.test_dataset) // 100) == 0:
                print(f"802ad Result: Processed {idx*100/len(self.test_dataset)}%  samples ...")
        self.run_res = pd.DataFrame(performance_list)
    
# sector level sweep baseline
class baseline_SectorSweep():
    def __init__(self, config_dataset, config_assumption, device):
        self.config = SimpleNamespace(
            dataset=copy.deepcopy(config_dataset),
            assumption=copy.deepcopy(config_assumption),
            generator=SimpleNamespace(
                type='uniform',
                M_act=4,
                N_act=4,
            )
        )
        max_rss_cache = cache_max_rss(config_dataset)
        self.max_rss_performance = max_rss_cache.performance
        _, test_dataset = load_datasets(self.config)
        self.test_dataset = test_dataset
        self.dataset_folder = test_dataset.dataset_path
        self.device = device
        self.generator = load_predefined_generator(self.config, device=device)
        self.dp = load_data_process(self.config, device)

        result_cache_folder = os.path.join('cache', 'SectorSweep', 'results')
        os.makedirs(result_cache_folder, exist_ok = True)
        self.result_path = os.path.join(result_cache_folder, namespace_to_filename(config_dataset, config_assumption, suffix='.csv'))
        if os.path.exists(self.result_path) == False:
            print(f"Sector Sweep Result: Result file not found at {self.result_path}, start running to collect...")
            self.run()
            self.performance = pd.merge(self.max_rss_performance, self.run_res, on="csi_path")
            self.performance["loss"] = self.performance["max_rss"] - self.performance["rss"]
            self.performance['angle_error'] = calculate_angle_error(self.performance['gt_phi'], self.performance['gt_theta'], self.performance['sector_sweep_phi'], self.performance['sector_sweep_theta'])
            self.performance.to_csv(self.result_path, index=False)
        self.performance = pd.read_csv(self.result_path)
        print(f"Sector Sweep Result is collected, and save to {self.result_path}")
        print("Sector Sweep Result is loaded")

    def generate(self, csi):
        weights, _ = self.generator.generate()
        rss = self.dp.generate_sample_rss(csi, weights)
        rss_max_idx = np.argmax(gpu_tensor_to_np(rss.squeeze(0)))
        sample_points = get_uniform_samples(self.generator.sample_num, self.generator.antenna_info.max_theta)
        sector_sweep_phi, sector_sweep_theta = sample_points[rss_max_idx]
        return np.max(get_tensor_db(rss)), sector_sweep_phi, sector_sweep_theta
    
    def predict_rss_at_gt(self, csi, gt_phi, gt_theta):
        """
        Predict RSS at ground truth direction.
        For sector sweep, find closest sector and scale by array size ratio.
        """
        weights, _ = self.generator.generate()
        rss = self.dp.generate_sample_rss(csi, weights)
        sample_points = get_uniform_samples(self.generator.sample_num, self.generator.antenna_info.max_theta)
        
        # Find closest sector to GT
        closest_idx = find_closest_direction_index(gt_phi, gt_theta, sample_points)
        rss_at_sector = rss.squeeze(0)[closest_idx]
        
        # Sector beam uses 4x4 array, while full array is 16x16
        # Scale by array size ratio: (16*16)/(4*4) = 16
        array_size_ratio = (self.config.dataset.M * self.config.dataset.N) / (self.generator.M_act * self.generator.N_act)
        predicted_rss = rss_at_sector * array_size_ratio
        
        return get_db(predicted_rss.item())
        
    def run(self):
        performance_list = []
        for idx, (csi, csi_file_path) in enumerate(self.test_dataset):
            csi = csi.unsqueeze(0).to(self.device)  # [1, M, N, F]
            rss, sector_sweep_phi, sector_sweep_theta = self.generate(csi)
            
            # Get GT direction and predict RSS at GT
            gt_row = self.max_rss_performance[self.max_rss_performance['csi_path'] == csi_file_path]
            if not gt_row.empty:
                gt_phi = gt_row['gt_phi'].values[0]
                gt_theta = gt_row['gt_theta'].values[0]
                rss_at_gt = self.predict_rss_at_gt(csi, gt_phi, gt_theta)
            else:
                raise ValueError(f"GT direction not found for {csi_file_path}")
            
            performance_list.append({
                "csi_path": csi_file_path,
                "rss": rss.item(),
                "sector_sweep_phi": sector_sweep_phi,
                "sector_sweep_theta": sector_sweep_theta,
                "rss_at_gt": rss_at_gt
            })
            if idx % max(1, len(self.test_dataset) // 100) == 0:
                print(f"Sector Sweep Result: Processed {idx*100/len(self.test_dataset)}%  samples ...")
        self.run_res = pd.DataFrame(performance_list)



class baseline_AgileLink():
    def __init__(self, config_dataset, config_assumption, device, num_sub_beam = 2):
        max_rss_cache = cache_max_rss(config_dataset)
        self.max_rss_performance = max_rss_cache.performance
        self.config = SimpleNamespace(
            dataset=copy.deepcopy(config_dataset),
            assumption=copy.deepcopy(config_assumption),
            generator=SimpleNamespace(
                type='uniform',
                M_act=config_dataset.M,
                N_act=config_dataset.N,
            )
        )
        _, test_dataset = load_datasets(self.config)
        self.test_dataset = test_dataset
        self.dataset_folder = test_dataset.dataset_path

        # if sample_num != None:
        #     self.config.assumption.sample_num = sample_num
        result_cache_folder = os.path.join('cache', 'AgileLink', 'results')
        self.result_path = os.path.join(result_cache_folder, namespace_to_filename(config_dataset, config_assumption, suffix='.csv'))
        # M=16,N=16,sub_beam=2, then beam number per epoch = 16
        assert (config_dataset.M == 16) and (config_dataset.N == 16) and (num_sub_beam == 2)

        
        self.num_sub_beam = num_sub_beam
        self.M, self.N = self.config.dataset.M, self.config.dataset.N
        self.bin_num_phi = self.config.dataset.M // (self.num_sub_beam)**2
        self.bin_num_theta = self.config.dataset.N // (self.num_sub_beam)**2
        self.device = device

        self.generator = load_predefined_generator(self.config, device=device)

        config_with_1_sample = copy.deepcopy(self.config)
        config_with_1_sample.assumption.sample_num = 1
        self.dp = load_data_process(config_with_1_sample, device) # for generate angle spectrum aim at ONE specific direction

        self.max_theta = self.config.dataset.max_theta


        os.makedirs(result_cache_folder, exist_ok = True)
        
        if os.path.exists(self.result_path) == False:
            print(f"AgileLink Result: Result file not found at {self.result_path}, start running to collect...")
            self.run()
            self.performance = pd.merge(self.max_rss_performance, self.run_res, on="csi_path")
            self.performance["loss"] = self.performance["max_rss"] - self.performance["rss"]
            # calculate_angle_error
            self.performance['angle_error'] = calculate_angle_error(self.performance['gt_phi'], self.performance['gt_theta'], self.performance['agilelink_phi'], self.performance['agilelink_theta'])
            self.performance.to_csv(self.result_path, index=False)
        self.performance = pd.read_csv(self.result_path)
        print(f"AgileLink Result is collected, and save to {self.result_path}")
        print("AgileLink Result is loaded")

    def run_per_csi(self, csi, gt_phi=None, gt_theta=None):
        angle_spectrum = self.dp.generate_query_rss(csi).reshape((80,20))
        start_time = time.time()
        
        # Pass GT info to run_agile_link to get the matrix-calculated score at GT grid
        (theta_opt, phi_opt), rss_pred_at_gt_raw = run_agile_link(
            angle_spectrum, self.M, self.N, 
            self.config.assumption.sample_num, 
            R = self.num_sub_beam,
            gt_phi=gt_phi, gt_theta=gt_theta
        )
        run_time = time.time() - start_time
        
        # Calculate physical RSS for the Best Prediction (Optimization result)
        weights_opt = self.generator.antenna_info.generate_toward_angles(thetas=[theta_opt], phis=[phi_opt])
        weights_opt = torch.from_numpy(weights_opt).to(self.device).view(1,1,self.M, self.N)  #  [batch_size(b), sample_num(s), M_tx, N_tx]
        rss = self.dp.generate_sample_rss(csi, weights_opt).item()

        # Use the raw matrix score as the "RSS at GT" (Algorithm's internal belief)
        # Note: This value is a product of energies (Soft Voting), not dBm.
        rss_at_gt = rss_pred_at_gt_raw

        return get_db(rss), phi_opt, theta_opt, run_time, rss_at_gt
    
    def run(self):
        res = []
        for i in range(len(self.test_dataset)):
            csi,csi_path = self.test_dataset[i]
            csi = csi.unsqueeze(0).to(self.device)
            
            # Get GT direction
            gt_row = self.max_rss_performance[self.max_rss_performance['csi_path'] == csi_path]
            if not gt_row.empty:
                gt_phi = gt_row['gt_phi'].values[0]
                gt_theta = gt_row['gt_theta'].values[0]
            else:
                raise ValueError(f"GT direction not found for {csi_path}")

            # Run Agile-Link
            rss, agilelink_phi, agilelink_theta, run_time, rss_at_gt = self.run_per_csi(csi, gt_phi=gt_phi, gt_theta=gt_theta)

            res.append({
                "csi_path":csi_path,
                "rss": rss,
                "agilelink_phi": agilelink_phi,
                "agilelink_theta": agilelink_theta,
                "agilelink_running_time": run_time,
                "rss_at_gt": rss_at_gt, # Stores the matrix-calculated score
            })
            if i % max(1, len(self.test_dataset) // 100) == 0:
                print(f"AgileLink Result: Processed {i*100/len(self.test_dataset)}%  samples ...")
        self.run_res = pd.DataFrame(res)
class cache_max_rss():
    def __init__(self, config_dataset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = SimpleNamespace(
            dataset=copy.deepcopy(config_dataset),
            assumption=SimpleNamespace(
                sample_num=64, # useless parameter
                angle_steps_theta=20,
                angle_steps_phi=80,
                array_factor_steps_theta=16,
                array_factor_steps_phi=64,
                angle_spectrum_length = 80*20,
            ))
        _, test_dataset = load_datasets(self.config)
        self.test_dataset = test_dataset
        self.dataset_folder = test_dataset.dataset_path
        self.device = device
        self.dp = load_data_process(self.config, device)
        result_cache_folder = os.path.join('cache', 'max_rss', 'results')
        os.makedirs(result_cache_folder, exist_ok = True)
        self.result_path = os.path.join(result_cache_folder, namespace_to_filename(config_dataset, config_dataset, suffix='.csv')) # assumption parameter is useless
        if os.path.exists(self.result_path) == False:
            print(f"Max RSS Result: Result file not found at {self.result_path}, start running to collect...")
            self.run()
            print(f"Max RSS Result is collected, and save to {self.result_path}")
        self.performance = pd.read_csv(self.result_path)
        print(f"Max RSS Result is loaded from {self.result_path}")

    def run(self):
        performance_list = []
        batch_size = 25
        
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,  
            drop_last=False
        )
        
        processed_count = 0
        with torch.no_grad():
            for batch_idx, (csi_batch, csi_path_batch) in enumerate(dataloader):
                csi_batch = csi_batch.to(self.device)  # [B, M, N, F]
                rss, phi_degrees, theta_degrees, _ = self.dp.generate_max_rss(csi_batch, return_direction = True)
                
                actual_batch_size = csi_batch.shape[0]
                
                for i in range(actual_batch_size):
                    csi_file_path = csi_path_batch[i]
                    rss_i = get_tensor_db(rss[i])
                    performance_list.append({
                        "csi_path": csi_file_path,
                        "max_rss": rss_i,
                        "gt_phi": phi_degrees[i].item(),
                        "gt_theta": theta_degrees[i].item()
                    })
                    processed_count += 1
                
                if batch_idx % max(1, len(dataloader) // 20) == 0:
                    progress = (processed_count / len(self.test_dataset)) * 100
                    print(f"Max RSS Result: Processed {progress:.1f}% ({processed_count}/{len(self.test_dataset)}) samples ...")
        
        df = pd.DataFrame(performance_list)
        df.to_csv(self.result_path, index=False)


def load_related_work(config_include_related_work):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    related_work_dict = {}
    for config in config_include_related_work:
        check_related_work_name(config.name)
        related_work_dict_ins = globals()[f"baseline_{config.name}"](config.dataset, config.assumption, device)
        related_work_dict[config.name] = related_work_dict_ins.performance
    return related_work_dict

if __name__ == "__main__":
    pass