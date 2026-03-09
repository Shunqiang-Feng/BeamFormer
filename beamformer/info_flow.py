# %%
from math import pi, log
from functools import wraps
import sys,os
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
import torch.nn.init as init

from .utils import scale_in_last_dim
# helpers
def normalize_weights_l2(weights):

    weights_np = np.asarray(weights, dtype=np.float64)

    norm = np.linalg.norm(weights_np)

    if norm == 0:
        print("Error: The input vector has zero norm, cannot be normalized.")
        return weights_np

    normalized_weights = weights_np / norm

    return normalized_weights




def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        seq_dropout_prob = 0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        queries = None,
        eff_layers = None,
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b = b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # structured dropout (as done in perceiver AR https://arxiv.org/abs/2202.07765)

        if self.training and self.seq_dropout_prob > 0.:
            data, mask = dropout_seq(data, mask, self.seq_dropout_prob)

        # cross attention only happens once for Perceiver IO

        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x

        # layers

        for layer_idx, (self_attn, self_ff) in enumerate(self.layers):
            if (eff_layers != None) and (layer_idx >= eff_layers):
                break
            if (eff_layers != None) and eff_layers > len(self.layers):
                print(f"You only have {len(self.layers)} layers!")
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        
        latents = self.decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        return self.to_logits(latents)

# Perceiver LM example


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
            raise ValueError(f"Unsupported estimator type: {estimator_config.type}")
        
        # Output layer
        self.output_layer = nn.Linear(estimator_config.dim, 1)  # Predicts a single value per token

        self._initialize_weights()  

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, sample_rss, sample_pos_enc, query_pos_enc, eff_layers = None):
        
        sample_rss_embedding = self.rss_bembedding(sample_rss.unsqueeze(-1)) 
        encoder_input = sample_rss_embedding + self.positional_encoding(sample_pos_enc)

        decoder_input = self.positional_encoding(query_pos_enc)
        if self.perceiver_io_flag:
            output = self.transformer(encoder_input, queries = decoder_input, eff_layers = eff_layers)  
        else:  
            output = self.transformer(encoder_input, decoder_input)
        output = self.output_layer(output).squeeze(-1)
        
        return scale_in_last_dim(output)

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple
from scipy.ndimage import zoom

# Set DejaVu Sans font, base font size and high resolution for academic standards
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300  # Set display resolution
plt.rcParams['savefig.dpi'] = 300  # Set save resolution

def gpu_tensor_to_np(tensor):
    """Convert GPU tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def interpolate_tensor(tensor, target_shape=(360, 90)):
    """
    Interpolate tensor from (80, 20) to target shape (360, 90) using bicubic interpolation
    
    Parameters:
    tensor: torch.Tensor or numpy array with shape (80, 20)
    target_shape: tuple, target shape for interpolation
    
    Returns:
    numpy array with target shape
    """
    # Convert to numpy if needed
    if isinstance(tensor, torch.Tensor):
        np_array = tensor.detach().cpu().numpy()
    else:
        np_array = tensor
    
    # Calculate zoom factors
    zoom_factors = (target_shape[0] / np_array.shape[0], 
                   target_shape[1] / np_array.shape[1])
    
    # Use bicubic interpolation (order=3)
    interpolated = zoom(np_array, zoom_factors, order=3, mode='nearest')
    
    return interpolated

def plot_multi_row_comparison(
    data_rows: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    col_title: str = "Normalized RSS",
    save_path: str = None,
    max_theta: float = 90,
    colormap: str = 'viridis',
    show_colorbar: bool = True,
    show_colorbar_label: bool = True,
    show_angle_lines: bool = False,
    show_peak_coordinates: bool = False,
    interpolate_to_shape: Tuple[int, int] = (360, 90),
    width_ratio: float = 2.0
):
    """
    
    Parameters:
    data_rows: List of tuples, each containing (gt_tensor, list_of_processed_tensors)
    col_title: str, colorbar title
    save_path: str, save path for the figure
    max_theta: float, maximum elevation angle range (degrees)
    colormap: str, matplotlib colormap name
    show_colorbar: bool, whether to show colorbar
    show_colorbar_label: bool, whether to show colorbar label
    show_angle_lines: bool, whether to show angle grid lines
    show_peak_coordinates: bool, whether to show peak coordinates
    interpolate_to_shape: tuple, target shape for interpolation (phi_bins, theta_bins)
    width_ratio: float, figure width ratio relative to base width (3.333 inches)
    """
    num_rows = len(data_rows)
    if num_rows == 0:
        print("Warning: data_rows is empty. Nothing to plot.")
        return
    
    # Check that all rows have exactly 8 processed tensors
    for row_idx, (gt_tensor, processed_tensors) in enumerate(data_rows):
        if len(processed_tensors) != 8:
            raise ValueError(f"Row {row_idx} must contain exactly 8 processed tensors.")
    
    # Figure dimensions (adjust width to accommodate extra spacing)
    base_width = 3.333  # MobiCom single column width
    fig_width = base_width * width_ratio * 1.08  # Increase width by 8% for extra spacing
    fig_height = fig_width * (num_rows * 0.125) * 1.03  # Compact height ratio
    
    # Create figure with custom gridspec for variable column widths
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create a gridspec with variable width ratios
    # GT column gets normal width, then there's a smaller spacer, then 8 layer columns get normal width
    width_ratios = [1.0, 0.15, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # GT, spacer, 8 layers
    gs = fig.add_gridspec(num_rows, 10, width_ratios=width_ratios)
    
    # Create axes array manually
    axes = []
    for row in range(num_rows):
        row_axes = []
        # GT column (index 0)
        ax_gt = fig.add_subplot(gs[row, 0], projection='polar')
        row_axes.append(ax_gt)
        
        # Skip spacer column (index 1)
        
        # Layer columns (indices 2-9)
        for col in range(2, 10):
            ax_layer = fig.add_subplot(gs[row, col], projection='polar')
            row_axes.append(ax_layer)
        
        axes.append(row_axes)
    
    # Convert to numpy array for easier indexing
    axes = np.array(axes)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Find global min/max for consistent color scaling across all plots
    all_data = []
    for gt_tensor, processed_tensors in data_rows:
        all_tensors = [gt_tensor] + processed_tensors
        for tensor in all_tensors:
            np_img = gpu_tensor_to_np(tensor)
            interpolated_img = interpolate_tensor(np_img, interpolate_to_shape)
            all_data.append(interpolated_img)
    
    all_data = np.array(all_data)
    # Process each row of data
    for row_idx, (gt_tensor, processed_tensors) in enumerate(data_rows):
        all_tensors = [gt_tensor] + processed_tensors
        
        # Process each tensor in the row
        for col_idx, tensor in enumerate(all_tensors):
            ax = axes[row_idx, col_idx]
            
            # Convert tensor to numpy and interpolate
            np_img = gpu_tensor_to_np(tensor)
            interpolated_img = interpolate_tensor(np_img, interpolate_to_shape)
            
            # Normalize using global min/max for consistency
            img_normalized = (interpolated_img - np.min(interpolated_img)) / (np.max(interpolated_img) - np.min(interpolated_img) + 1e-8)
            
            # Create polar coordinate grid
            phi_bins, theta_bins = interpolated_img.shape
            phi_rad = np.linspace(0, 2*np.pi, phi_bins)
            theta_norm = np.linspace(0, 1, theta_bins)
            THETA, R = np.meshgrid(phi_rad, theta_norm)
            
            # Create contour plot
            levels = np.linspace(0, 1, 20)
            im = ax.contourf(THETA, R, img_normalized.T, levels=levels, 
                           cmap=colormap, alpha=1, vmin=0, vmax=1)
            
            # Set polar coordinate parameters
            ax.set_ylim(0, 1)
            ax.set_theta_zero_location('S')
            ax.set_theta_direction(1)
            
            # Minimal grid and labels for compact display
            if show_angle_lines:
                ax.set_thetagrids([0, 90, 180, 270], alpha=0.3)
                ax.set_rgrids([0.5, 1.0], labels=['45°', '90°'], fontsize=8, alpha=0.5)
                ax.grid(True, alpha=0.2, linewidth=0.3)
            else:
                ax.set_thetagrids([])
                ax.set_rgrids([])
                ax.grid(False)
            

            # Show peak coordinates if requested
            if show_peak_coordinates:
                max_idx = np.unravel_index(img_normalized.argmax(), img_normalized.shape)
                peak_phi_idx = max_idx[0]
                peak_theta_idx = max_idx[1]
                
                phi_values = np.linspace(0, 360, phi_bins)
                theta_values = np.linspace(0, max_theta, theta_bins)
                peak_phi_deg = phi_values[peak_phi_idx]
                peak_theta_deg = theta_values[peak_theta_idx]
                
                peak_phi_rad = np.radians(peak_phi_deg)
                peak_r = peak_theta_deg / max_theta
                
                ax.plot(peak_phi_rad, peak_r, 'r*', markersize=3)
            
            # Remove tick labels for cleaner look
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
    # Add column labels at bottom (only for last row)
    column_labels = ["Ground\nTruth", "1\nLayer", "2\nLayers", "3\nLayers", 
                    "4\nLayers", "5\nLayers", "6\nLayers", "7\nLayers", "8\nLayers"]
    
    for col_idx, label in enumerate(column_labels):
        axes[-1, col_idx].text(0.5, -0.15, label, transform=axes[-1, col_idx].transAxes, 
                              ha='center', va='top', fontsize=12, 
                              weight='bold' if col_idx == 0 else 'normal')
    
    # Add row labels on the left
    for row_idx in range(num_rows):
        axes[row_idx, 0].text(-0.35, 0.5, f'Sample {row_idx+1}', 
                             transform=axes[row_idx, 0].transAxes,
                             rotation=90, va='center', ha='center', 
                             fontsize=12, weight='bold')
    

    
    # Add unified colorbar if requested
    if show_colorbar:
        # Position colorbar on the right side
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        if show_colorbar_label:
            cbar.set_label(col_title, fontsize=10, labelpad=10)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        cbar.ax.tick_params(labelsize=8)
    
    # Tight layout with adjusted spacing
    plt.subplots_adjust(left=0.08, right=0.9 if show_colorbar else 0.98, 
                       top=0.95, bottom=0.15, 
                       wspace=0.02, hspace=0.03)  # Reduced wspace since spacing is handled by gridspec
    
    # Save or show the figure
    if save_path:
        output_dir = os.path.dirname(os.path.abspath(save_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='white', edgecolor='none')
    else:
        plt.show()
    
    plt.close(fig)
