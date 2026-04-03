import torch
import torch.nn as nn
from .utils import antenna_info, get_uniform_samples
import numpy as np


def transform_weights(B):

    if not torch.all((B >= -1) & (B <= 1)):
        raise ValueError("All elements in B must be in the range [-1, 1].")

    active_antenna = (B >= 0) 
    weights_output = torch.zeros_like(B, dtype= torch.complex64)
    weights_output[active_antenna] = torch.exp(2 * torch.pi * 1j * B[active_antenna])
    
    return weights_output,active_antenna
    

class ParametricGenerator(nn.Module):
    def __init__(self, channel_number, M_base, N_base, initial_value=None):
        super().__init__()
        self.channel_number = channel_number
        self.M_base = M_base
        self.N_base = N_base
        self.input_dim = channel_number * M_base * N_base

        if initial_value is None:
            self.weights = nn.Parameter(torch.empty(1, self.input_dim))
            nn.init.uniform_(self.weights, -1, 1)
        else:
            # 加载 numpy 文件
            init_array = np.load(initial_value)  # e.g., "uniform_4x4_hidden_variable.npy"
            init_tensor = torch.from_numpy(init_array).float()

            # 确保尺寸匹配
            assert init_tensor.numel() == self.input_dim, \
                f"Loaded tensor has {init_tensor.numel()} elements, expected {self.input_dim}"

            # reshape 并注册为可训练参数
            self.weights = nn.Parameter(init_tensor.view(1, self.input_dim))
    
    def forward(self, z=None):
        batch_size = z.size(0) if z is not None else 1
        x = self.weights.repeat(batch_size, 1)  # 保持 batch 维度
        x = torch.tanh(x) # necessary
        x = x.view(batch_size, self.channel_number, self.M_base, self.N_base)
        return x

    def generate(self, z=None):
        raw_weights = self.forward(z)
        weight, active_antenna = transform_weights(raw_weights)
        return weight, active_antenna



    
class PredefinedGenerator():
    def __init__(self, M_act, N_act, sample_mode, sample_num, M_base, N_base, start_freq,end_freq,angle_steps_theta, angle_steps_phi, freq_num, d_row, d_col, max_theta, device):
        self.M_act = M_act
        self.N_act = N_act
        self.M_base = M_base
        self.N_base = N_base

        self.antenna_info = antenna_info(start_freq = start_freq,end_freq = end_freq,freq_num = freq_num,M=M_act,N=N_act,d_row = d_row, d_col=d_col, max_theta = max_theta)
        self.sample_mode = sample_mode
        self.sample_num = sample_num
        self.angle_steps_theta = angle_steps_theta
        self.angle_steps_phi = angle_steps_phi
        self.device = device

    def _generate_query_weights(self):
        # return [M, N, phi_steps*theta_steps], [16, 16, 1600]
        weights = self.antenna_info.generate_angles(self.angle_steps_theta,self.angle_steps_phi)
        weights = torch.from_numpy(weights)
        return weights.reshape(self.M_act, self.N_act, self.angle_steps_theta*self.angle_steps_phi)
    
    def _generate_small_shape(self):
        # return [M, N, sample_num]
        if self.sample_mode == "random":
            random_phase = 2 * torch.pi * torch.rand(self.M_act, self.N_act, self.sample_num)
            weights = torch.exp(1j * random_phase)
        elif self.sample_mode == "uniform" or self.sample_mode == "fixed" :
            sample_points = get_uniform_samples(self.sample_num, self.antenna_info.max_theta)
            weights = torch.zeros(self.M_act, self.N_act, self.sample_num, dtype=torch.complex64)
            for i, (phi, theta) in enumerate(sample_points):
                weight_for_angle = self.antenna_info.generate_toward_angles([theta], [phi])
                weights[:, :, i] = torch.from_numpy(weight_for_angle).reshape(self.M_act, self.N_act)
        else:
            raise ValueError("Not valid sample_mode")
        return weights / torch.abs(weights)
    
    
    def generate(self, batch_size = 1):
        # return large_shape: [batch_size, sample_num, M_base, N_base], active_antenna: [batch_size, sample_num, M_base, N_base]
        small_shape = self._generate_small_shape()
        # create large shape zero complex tensor
        large_shape = torch.zeros(self.M_base, self.N_base, self.sample_num, dtype=torch.complex64)
        large_shape[:self.M_act, :self.N_act, :] = small_shape
        active_antenna = (abs(large_shape) > 0).float()
        large_shape = large_shape.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 3, 1, 2)
        active_antenna = active_antenna.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 3, 1, 2)
        return large_shape.to(self.device),active_antenna.to(self.device)  # weights, active_antenna       

    def generate_weights_toward_angles(self, phi_theta_turple, one_direction_only = False):
        # return [sample_num ,M, N]
        if one_direction_only:
            phi_theta_turple = [phi_theta_turple]
        else:
            if len(phi_theta_turple) != self.sample_num:
                raise ValueError(f"the length of phi_theta_turple should be {self.sample_num}, but got {len(phi_theta_turple)}")
        large_shape = torch.zeros(self.M_base, self.N_base, len(phi_theta_turple), dtype=torch.complex64)
        for i in range(len(phi_theta_turple)):
            phi, theta = phi_theta_turple[i]
            weights = self.antenna_info.generate_toward_angles(thetas=[theta], phis=[phi])
            weights = torch.from_numpy(weights)
            large_shape[:self.M_act, :self.N_act, i] = weights.reshape(self.M_act, self.N_act)

        return large_shape.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # [1, sample_num, M_base, N_base] 
        


def load_predefined_generator(config, device):
    ds = config.dataset
    dg = config.generator
    return PredefinedGenerator(
        M_act = dg.M_act,
        N_act = dg.N_act,
        sample_mode = dg.type,
        sample_num = config.assumption.sample_num,
        M_base = ds.M,
        N_base = ds.N,
        start_freq = ds.start_freq,
        end_freq = ds.end_freq,
        angle_steps_theta = config.assumption.angle_steps_theta,
        angle_steps_phi = config.assumption.angle_steps_phi,
        freq_num= ds.freq_num,
        d_col=ds.d_col,
        d_row=ds.d_row,
        max_theta = ds.max_theta,
        device=device,
    )