import torch,math
from .utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.init as init
from .perceiver_io import PerceiverIO


    
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

def get_scheduler_by_type(optimizer, warmup_steps, total_steps, num_cycles=0.5, last_epoch: int = -1):
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, num_cycles=num_cycles, last_epoch=last_epoch)


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

    def forward(self, sample_rss, sample_pos_enc, query_pos_enc):
        
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
    def forward(self, sample_rss, sample_pos_encoding, query_pos_encoding):
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
