import torch,math
import numpy as np
from .utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.init as init
from .perceiver_io import PerceiverIO


class ArrayAdapter(nn.Module):
    """Projects beam pattern (array factor) from large input_dim to output_dim."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


    
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 3.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 3.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a 3 & half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_step_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    Scheduler with:
    - Linear warmup for `num_warmup_steps`
    - Then StepLR: every 1/4 of remaining steps, lr decays by factor of 10
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Remaining steps after warmup
            remaining = num_training_steps - num_warmup_steps
            step_in_decay = current_step - num_warmup_steps
            decay_stage = step_in_decay * 6 // max(1, remaining)  # 0~3
            return 1.0 / (10 ** decay_stage)

    return LambdaLR(optimizer, lr_lambda, last_epoch) 

def get_flat_cosine_schedule(
    optimizer: torch.optim.Optimizer, 
    num_training_steps: int, 
    num_flat_steps: int, 
    last_epoch: int = -1
):

    if num_flat_steps >= num_training_steps:
        raise ValueError("num_flat_steps must be less than num_training_steps.")

    def lr_lambda(current_step: int):
        if current_step < num_flat_steps:
            return 1.0

        progress = float(current_step - num_flat_steps) / float(max(1, num_training_steps - num_flat_steps))
        
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_scheduler_by_type(scheduler_type, optimizer, warmup_steps, total_steps, flat_steps = 0, num_cycles=0.5, last_epoch: int = -1):
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, num_cycles=num_cycles, last_epoch = last_epoch)
    elif scheduler_type == "step":
        return get_step_lr_with_warmup(optimizer, warmup_steps, total_steps, last_epoch = last_epoch)
    elif scheduler_type == "flat_cosine":
        if flat_steps == 0:
            flat_steps = total_steps // 2
            print(f"Warning: flat_steps not provided for 'flat_cosine' scheduler. Defaulting to half of total_steps: {flat_steps}")
        return get_flat_cosine_schedule(optimizer, total_steps, flat_steps, last_epoch=last_epoch)
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class TransformerModel(nn.Module):
    # def __init__(self, dim, num_encoder_layers, num_decoder_layers, nhead, dim_feedforward, transformer_type):
    def __init__(self, estimator_config):
        # For Transformer: dim, num_encoder_layers, num_decoder_layers, nhead, dim_feedforward, transformer_type
        # For Perceiver IO: dim, depth, queries_dim, num_latents, latent_dim, cross_heads, latent_heads, cross_dim_head, latent_dim_head, decoder_ff, transformer_type
        super(TransformerModel, self).__init__()
        # self.dim = estimator_config.dim

        # else:
        #     self.mask_token = nn.Parameter(torch.rand(1, mask_token_num, dim))
        self.rss_bembedding = nn.Linear(1, estimator_config.dim)
        

        self.positional_encoding = nn.Sequential(
                        nn.Linear(estimator_config.array_factor_len, estimator_config.dim_feedforward),  
                        nn.ReLU(),           
                        nn.Linear(estimator_config.dim_feedforward, estimator_config.dim_feedforward),  
                        nn.ReLU(),           
                        nn.Linear(estimator_config.dim_feedforward, estimator_config.dim)    
                    )
        # Transformer module
        self.perceiver_io_flag = False
        if estimator_config.type.lower() == "perceiver_io":
            self.transformer = PerceiverIO(depth= estimator_config.depth,
            dim=estimator_config.dim, queries_dim=estimator_config.queries_dim, num_latents=estimator_config.num_latents, latent_dim=estimator_config.latent_dim, cross_heads= estimator_config.cross_heads, latent_heads=estimator_config.latent_heads, cross_dim_head=estimator_config.cross_dim_head, latent_dim_head=estimator_config.latent_dim_head, decoder_ff=estimator_config.decoder_ff,) # logits_dim = 1024)
            self.perceiver_io_flag = True
        else:
            raise ValueError(f"Unsupported estimator type: {estimator_config.type.lower()}")
        
        # Output layer
        self.output_layer = nn.Linear(estimator_config.dim, 1)  # Predicts a single value per token

        self._initialize_weights() 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, sample_rss, sample_pos_enc, query_pos_enc, grid_indices=None):

        sample_rss_embedding = self.rss_bembedding(sample_rss.unsqueeze(-1))
        encoder_input = sample_rss_embedding + self.positional_encoding(sample_pos_enc)

        decoder_input = self.positional_encoding(query_pos_enc)
        if self.perceiver_io_flag:
            output = self.transformer(encoder_input, queries = decoder_input)  
        else:  
            raise ValueError("Not Our Model")
        output = self.output_layer(output).squeeze(-1)
        
        return scale_in_last_dim(output)

class FastTransformerModel(TransformerModel):
    @torch.no_grad()
    def forward(self, sample_rss, sample_pos_encoding, query_pos_encoding, grid_indices=None):
        sample_rss_embedding = self.rss_bembedding(sample_rss.unsqueeze(-1))
        encoder_input = sample_rss_embedding + sample_pos_encoding
        decoder_input = query_pos_encoding

        if self.perceiver_io_flag:
            output = self.transformer(encoder_input, queries=decoder_input)
        else:
            raise ValueError("Not Our Model")

        output = self.output_layer(output).squeeze(-1)
        return scale_in_last_dim(output)

    @torch.no_grad()
    def prepare_positional_encoding(self, sample_pos_enc, query_pos_enc):

        sample_pos_encoding = self.positional_encoding(sample_pos_enc)
        query_pos_encoding = self.positional_encoding(query_pos_enc)
        return sample_pos_encoding, query_pos_encoding


class AmplitudeRecoveryNetwork(nn.Module):
    def __init__(self, sample_num: int = 64, d_model = 512):
        super(AmplitudeRecoveryNetwork, self).__init__()
        self.patch_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, sample_num, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [B, sample_num]
        x = self.patch_embed(x.unsqueeze(-1)) # [B, sample_num, d_model]
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.head(x).squeeze()  # output: B*sample_num


# ---------------------------------------------------------------------------
# UNet Baseline: sparse beam spectrum → angle spectrum
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Double convolution block: (Conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetModel(nn.Module):
    """
    UNet baseline for beam spectrum completion.

    Input:  Uniformly sampled beam spectrum with zeros at unsampled positions.
            Shape: (B, phi*theta) flattened, will be reshaped to (B, 1, phi, theta)
    Output: Complete angle spectrum, shape: (B, phi*theta)
    """

    def __init__(self, estimator_config):
        super().__init__()
        phi   = estimator_config.angle_steps_phi
        theta = estimator_config.angle_steps_theta
        C     = getattr(estimator_config, 'unet_base_channels', 64)

        self.phi, self.theta = phi, theta

        self.enc1 = _DoubleConv(1, C)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = _DoubleConv(C, C * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = _DoubleConv(C * 2, C * 4)
        self.pool3 = nn.MaxPool2d((2, 1))

        self.enc4 = _DoubleConv(C * 4, C * 8)
        self.pool4 = nn.MaxPool2d((2, 1))

        self.bottleneck = _DoubleConv(C * 8, C * 16)

        self.up4 = nn.ConvTranspose2d(C * 16, C * 8, kernel_size=(2, 1), stride=(2, 1))
        self.dec4 = _DoubleConv(C * 16, C * 8)

        self.up3 = nn.ConvTranspose2d(C * 8, C * 4, kernel_size=(2, 1), stride=(2, 1))
        self.dec3 = _DoubleConv(C * 8, C * 4)

        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(C * 4, C * 2)

        self.up1 = nn.ConvTranspose2d(C * 2, C, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(C * 2, C)

        self.out_conv = nn.Conv2d(C, 1, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    @torch.no_grad()
    def prepare_positional_encoding(self, sample_pos_enc, query_pos_enc):
        """Passthrough: UNet does not use positional encoding."""
        return sample_pos_enc, query_pos_enc

    def forward(self, sparse_spectrum):
        B = sparse_spectrum.shape[0]
        x = sparse_spectrum.view(B, 1, self.phi, self.theta)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        out = out.view(B, -1)

        return scale_in_last_dim(out)


# ---------------------------------------------------------------------------
# Helper: map uniform sample indices to angle grid positions
# ---------------------------------------------------------------------------

def compute_uniform_to_grid_indices(sample_num, max_theta, angle_steps_theta, angle_steps_phi):
    """
    Compute the mapping from uniform (golden-ratio) sample indices to the
    flattened angle-grid positions used by generate_query_rss / UNetModel.

    Returns:
        torch.LongTensor of shape (sample_num,)
    """
    sample_points = get_uniform_samples(sample_num, max_theta)

    thetas_grid = np.linspace(0, max_theta, angle_steps_theta)
    phis_grid   = np.linspace(0, 360, angle_steps_phi)

    indices = []
    for phi_val, theta_val in sample_points:
        theta_idx = int(np.argmin(np.abs(thetas_grid - theta_val)))
        phi_diffs = np.abs(phis_grid - phi_val)
        phi_diffs = np.minimum(phi_diffs, 360.0 - phi_diffs)
        phi_idx   = int(np.argmin(phi_diffs))
        flat_idx  = phi_idx * angle_steps_theta + theta_idx
        indices.append(flat_idx)

    return torch.LongTensor(indices)


# ---------------------------------------------------------------------------
# CNN (UNet) wrapper: adapts standard estimator interface → sparse_spectrum
# ---------------------------------------------------------------------------

class CNNGeneratorModel(nn.Module):
    """
    Wrapper around UNetModel that converts the standard estimator interface
        forward(sample_rss, sample_pos_enc, query_pos_enc, grid_indices=None)
    into the sparse-spectrum input expected by UNetModel.

    When grid_indices is None, falls back to pre-computed uniform-sampling buffer.
    """

    def __init__(self, estimator_config):
        super().__init__()
        self.unet = UNetModel(estimator_config)

        indices = compute_uniform_to_grid_indices(
            sample_num        = estimator_config.sample_num,
            max_theta         = estimator_config.max_theta,
            angle_steps_theta = estimator_config.angle_steps_theta,
            angle_steps_phi   = estimator_config.angle_steps_phi,
        )
        self.register_buffer('sample_indices', indices)

    @torch.no_grad()
    def prepare_positional_encoding(self, sample_pos_enc, query_pos_enc):
        return sample_pos_enc, query_pos_enc

    def forward(self, sample_rss, sample_pos_enc, query_pos_enc, grid_indices=None):
        sparse_spectrum = self._rss_to_sparse_spectrum(sample_rss, grid_indices)
        return self.unet(sparse_spectrum)

    def _rss_to_sparse_spectrum(self, sample_rss, grid_indices=None):
        B = sample_rss.shape[0]
        L = self.unet.phi * self.unet.theta
        sparse = torch.zeros(B, L, device=sample_rss.device, dtype=sample_rss.dtype)
        idx = (grid_indices if grid_indices is not None else self.sample_indices)
        idx = idx.to(device=sample_rss.device)
        idx = idx.unsqueeze(0).expand(B, -1)
        sparse.scatter_(1, idx, sample_rss)
        return sparse


# ---------------------------------------------------------------------------
# Baseline: MLP Auto-Encoder estimator
# ---------------------------------------------------------------------------

class MLPAutoEncoderModel(nn.Module):
    """
    MLP-based Auto-Encoder estimator (baseline).

    Encoder: per-sample MLP (rss + pos_enc) → global max-pool → latent vector.
    Decoder: symmetric MLP latent → full angle spectrum.

    Interface matches TransformerModel:
        forward(sample_rss, sample_pos_enc, query_pos_enc, grid_indices=None)
            → (normalized_spectrum, scale)    [query_pos_enc and grid_indices are ignored]
    """

    def __init__(self, estimator_config):
        super().__init__()
        L          = estimator_config.array_factor_len
        output_len = estimator_config.angle_steps_phi * estimator_config.angle_steps_theta
        dim        = estimator_config.dim
        h1         = dim // 2

        self.input_proj = nn.Linear(1 + L, h1)
        self.enc_fc1    = nn.Linear(h1, dim)
        self.enc_fc2    = nn.Linear(dim, h1)

        self.dec_fc1 = nn.Linear(h1, dim)
        self.dec_fc2 = nn.Linear(dim, dim)
        self.out_fc  = nn.Linear(dim, output_len)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    @torch.no_grad()
    def prepare_positional_encoding(self, sample_pos_enc, query_pos_enc):
        """Passthrough: MLP-AE takes raw pos enc inside forward."""
        return sample_pos_enc, query_pos_enc

    def forward(self, sample_rss, sample_pos_enc, query_pos_enc, grid_indices=None):
        x = torch.cat([sample_rss.unsqueeze(-1), sample_pos_enc], dim=-1)
        x = torch.relu(self.input_proj(x))
        x = torch.relu(self.enc_fc1(x))
        x = torch.relu(self.enc_fc2(x))

        x = x.max(dim=1).values

        x   = torch.relu(self.dec_fc1(x))
        x   = torch.relu(self.dec_fc2(x))
        out = self.out_fc(x)
        return scale_in_last_dim(out)
