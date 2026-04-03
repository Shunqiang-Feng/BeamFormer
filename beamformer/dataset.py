import torch,os
from .utils import antenna_info, beamform_complex_single_f_4_wg, read_csi_file_to_torch, normalize_weights, add_thermal_noise
from torch.utils.data import Dataset
import torch

def check_tensor_dim(tensor, expect_dim):
    if tensor.dim() != expect_dim:
        raise ValueError(f"Tensor dimension is {tensor.dim()}, but expected 4.")

class data_process_4_weight_generator():
    def __init__(self,  device , M_tx,N_tx,M_rx,N_rx,mode,
                 angle_steps_theta, angle_steps_phi, array_factor_steps_theta, array_factor_steps_phi,
                 d_row , d_col,
                 start_freq, end_freq, freq_num,
                 max_theta):
        # self.src_folder_path = src_folder_path
        self.angle_steps_theta = angle_steps_theta
        self.angle_steps_phi = angle_steps_phi
        self.array_factor_steps_theta = array_factor_steps_theta
        self.array_factor_steps_phi = array_factor_steps_phi
        self.freq_num = freq_num
        # self.raw_csi_paths = [os.path.join(src_folder_path, f) for f in os.listdir(src_folder_path) if f.endswith('.mat')]
        self.antenna_info_tx = antenna_info(start_freq = start_freq,end_freq = end_freq,freq_num = freq_num,M=M_tx,N=N_tx, d_col= d_col, d_row= d_row, max_theta= max_theta)
        self.antenna_info_rx = antenna_info(start_freq = start_freq,end_freq = end_freq,freq_num = freq_num,M=M_rx,N=N_rx, d_col= d_col, d_row= d_row, max_theta= max_theta)
        self.mode = mode
        # self.sample_mode = sample_mode
        self.device = device
        if mode == "rx_act1":
            self.M, self.N = M_tx, N_tx
            self.antenna_info = self.antenna_info_tx
            self.antenna_info_act1 = self.antenna_info_rx
        elif mode == "tx_act1":
            self.M, self.N = M_rx, N_rx
            self.antenna_info = self.antenna_info_rx
            self.antenna_info_act1 = self.antenna_info_tx
        else:
            raise ValueError("No Valid Mode Input")
        self.mid_lambda = self.antenna_info.mid_wavelen


    def _generate_act1_weights(self, K):
        # return [M*N, K]
        weights = self.antenna_info_act1.act_1st_antenna_vector(K)
        return torch.from_numpy(weights).to(self.device)
    
    def _generate_rss(self, csi_mat, weights):
        # csi_mat: [batch_size(b), freq(f), rx, tx]
        # weights: [batch_size(b), sample_num(s), t_or_r]
        # return [batch_size, sample_num]
        check_tensor_dim(csi_mat, 4)
        check_tensor_dim(weights, 3)

        batch_size = csi_mat.shape[0]
        act1_weights = self._generate_act1_weights(1).squeeze() # [t_or_r]
        act1_weights = act1_weights.view(1, 1, act1_weights.shape[0]).repeat(batch_size, weights.shape[1], 1) # [batch_size(b), sample_num(s), t_or_r]
        
        act1_weights = normalize_weights(act1_weights)
        weights = normalize_weights(weights)
        if self.mode == "rx_act1":
            return beamform_complex_single_f_4_wg(act1_weights, csi_mat, weights)
        elif self.mode == "tx_act1":
            return beamform_complex_single_f_4_wg(weights, csi_mat, act1_weights)
    
    def generate_sample_rss(self, csi_mat, sample_weights):
        # csi_mat: [batch_size(b), freq(f), rx, tx]
        # weights: [batch_size(b), sample_num(s), M_tx, N_tx]
        # ensure sample_weights share same precision with csi_mat
        sample_num = sample_weights.shape[1]
        batch_size = csi_mat.shape[0]
    
        sample_weights = sample_weights.to(dtype = csi_mat.dtype)
        sample_weights = sample_weights.view(batch_size, sample_num,-1)
        # sample_weights = sample_weights.reshape(batch_size, sample_num,-1)
        rss = self._generate_rss(csi_mat, sample_weights) # rss: [batch_size, sample_num]
        return rss



    def generate_max_rss(self, csi_mat, return_direction=False):
        batch_size = csi_mat.shape[0]
        query_num = self.antenna_info.max_theta * 360
        theta_steps = self.antenna_info.max_theta  # 例如 90
        phi_steps = 360
        weights = self.antenna_info.generate_angles(self.antenna_info.max_theta, 360)
        weights = torch.from_numpy(weights).reshape(self.M, self.N, query_num)
        weights = weights.unsqueeze(0).repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2).to(self.device)
        
        weights = weights.view(batch_size, query_num, -1)  # batch_size, query_num, M*N
        
        rss = self._generate_rss(csi_mat, weights)  # return [batch_size, query_num]
        
        max_rss, max_indices = rss.max(dim=1)
        
        if not return_direction:
            return max_rss
        
        theta_indices = max_indices % theta_steps  
        phi_indices = max_indices // theta_steps   
        
        phi_degrees = phi_indices.float() * (360.0 / phi_steps) 
        theta_degrees = theta_indices.float() * (self.antenna_info.max_theta / theta_steps)  
        
        angle_spectrum = rss.view(batch_size, phi_steps, theta_steps)
        
        return max_rss, phi_degrees, theta_degrees, angle_spectrum

    def generate_query_weights(self, batch_size):
        # target: [batch_size, sample_num, Tx_M, Tx_N]
        # Generate weights using antenna_info and convert to PyTorch tensor
        weights = self.antenna_info.generate_angles(self.angle_steps_theta, self.angle_steps_phi) # return np [M, N, phi_steps, theta_steps]
        weights = torch.from_numpy(weights).reshape(self.M, self.N, self.angle_steps_theta*self.angle_steps_phi) # M,N,sample_num
        weights = weights.unsqueeze(0).repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2)
        return weights.to(self.device)
    
    def generate_query_rss(self, csi_mat):
        batch_size = csi_mat.shape[0]
        query_num = self.angle_steps_theta*self.angle_steps_phi # 1600
        query_weights = self.generate_query_weights(batch_size).view(batch_size, query_num,-1)
        # print(f"CSI_MAT dtype: {csi_mat.dtype}, query_weights dtype: {query_weights.dtype}")
        return self._generate_rss(csi_mat, query_weights)
    
    def generate_sample_position_encoding(self, sample_weights):
        """Generate array-factor position encoding for samples. Returns [batch_size, sample_num, enc_dim]."""
        batch_size, sample_num, _, _ = sample_weights.shape
        sample_weights_reshaped = sample_weights.reshape(batch_size * sample_num,
                                                         self.M, self.N).permute(1, 2, 0)
        AF_magnitude, _ = self.antenna_info.get_af_vector(
            weights=sample_weights_reshaped,
            theta_steps=self.array_factor_steps_theta,
            phi_steps=self.array_factor_steps_phi,
            device=self.device)
        AF_magnitude = AF_magnitude.reshape(batch_size, sample_num, -1)
        return AF_magnitude.to(device=self.device, dtype=torch.float64)

    def generate_query_position_encoding(self, batch_size):
        """Generate array-factor position encoding for query points. Returns [batch_size, query_num, enc_dim]."""
        query_num = self.angle_steps_theta * self.angle_steps_phi
        query_weights = self.generate_query_weights(batch_size=batch_size)
        query_weights = query_weights.reshape(batch_size * query_num, self.M, self.N).permute(1, 2, 0)
        AF_magnitude, _ = self.antenna_info.get_af_vector(
            weights=query_weights,
            theta_steps=self.array_factor_steps_theta,
            phi_steps=self.array_factor_steps_phi,
            device=self.device)
        AF_magnitude = AF_magnitude.reshape(batch_size, query_num, -1)
        return AF_magnitude.to(self.device)


class csi_dataset(Dataset):
    def __init__(self, src_folder, M_tx = 16, N_tx = 16, M_rx = 2, N_rx = 1, add_noise = False, snr_min = None, subcarrier_bw = 240e3):
        if src_folder and os.path.isdir(src_folder):
            self.dataset_path = sorted([os.path.join(src_folder, i) for i in os.listdir(src_folder)])
        else:
            raise ValueError(f"Invalid or missing directory: {src_folder}")

        self.M_tx = M_tx
        self.N_tx = N_tx
        self.M_rx = M_rx
        self.N_rx = N_rx

        self.subcarrier_bw = subcarrier_bw
        self.add_noise = add_noise
        self.snr_min = snr_min

    def __getitem__(self, idx):
        csi_mat = read_csi_file_to_torch(self.dataset_path[idx], M_tx=self.M_tx, N_tx=self.N_tx, M_rx=self.M_rx, N_rx=self.N_rx)
        if self.add_noise:
            csi_mat = add_thermal_noise(csi_mat, subcarrier_bw = self.subcarrier_bw, snr_db=self.snr_min)
        return csi_mat, self.dataset_path[idx]

    def __len__(self):
        return len(self.dataset_path)
    

def load_datasets(c):
    def _make_csi_dataset(path):
        return csi_dataset(
            path,
            M_tx=c.dataset.M_tx,
            N_tx=c.dataset.N_tx,
            M_rx=c.dataset.M_rx,
            N_rx=c.dataset.N_rx,
            add_noise= c.dataset.add_noise,
            snr_min = c.dataset.snr_min,
            subcarrier_bw=c.dataset.subcarrier_spacing,
        )
    train_dataset = _make_csi_dataset(c.dataset.train_data_path)
    test_dataset = _make_csi_dataset(c.dataset.test_data_path)

    return train_dataset, test_dataset


def load_data_process(config, device):
    dp = data_process_4_weight_generator(
        device=device,
        M_tx = config.dataset.M_tx,
        N_tx = config.dataset.N_tx,
        M_rx = config.dataset.M_rx,
        N_rx = config.dataset.N_rx,
        mode = config.dataset.mode,
        angle_steps_theta = config.assumption.angle_steps_theta,
        angle_steps_phi = config.assumption.angle_steps_phi,
        array_factor_steps_theta = config.assumption.array_factor_steps_theta,
        array_factor_steps_phi = config.assumption.array_factor_steps_phi,
        d_row = config.dataset.d_row,
        d_col = config.dataset.d_col,
        start_freq = config.dataset.start_freq,
        end_freq = config.dataset.end_freq,
        freq_num = config.dataset.freq_num,
        max_theta = config.dataset.max_theta,
    )
    return dp
