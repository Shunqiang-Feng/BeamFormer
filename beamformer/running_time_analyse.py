# %%
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.init as init
from perceiver_pytorch import PerceiverIO
import pandas as pd
import numpy as np
import time
import beamformer.utils as utils
from types import SimpleNamespace
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from configs.submodules.estimator import PerceiverIO as PerceiverIO_config

# Allow output directory to be overridden via environment variable.
# Default: eval_results/running_time_comparison/cached
_OUTPUT_DIR = os.environ.get(
    'RT_OUTPUT_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'eval_results', 'running_time_comparison', 'cached')
)
os.makedirs(_OUTPUT_DIR, exist_ok=True)

class TransformerModel(nn.Module):
    def __init__(self, estimator_config):
        # For Transformer: dim, num_encoder_layers, num_decoder_layers, nhead, dim_feedforward, transformer_type
        # For Perceiver IO: dim, depth, queries_dim, num_latents, latent_dim, cross_heads, latent_heads, cross_dim_head, latent_dim_head, decoder_ff, transformer_type
        super(TransformerModel, self).__init__()
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
        elif estimator_config.type.lower() == "standard":
            self.transformer = nn.Transformer(
                d_model=estimator_config.dim,
                nhead=estimator_config.nhead,
                num_encoder_layers=estimator_config.num_encoder_layers,
                num_decoder_layers=estimator_config.num_decoder_layers,
                dim_feedforward=estimator_config.dim_feedforward,
                batch_first=True,
  
            )
            self.perceiver_io_flag = False
        else:
            raise ValueError(f"Unsupported estimator type: {estimator_config.type}")
        
        self.output_layer = nn.Linear(estimator_config.dim, 1) 

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
            output = self.transformer(
                src=encoder_input, 
                tgt=decoder_input
            )

        output = self.output_layer(output).squeeze(-1)
        
        return utils.scale_in_last_dim(output)

class FastTransformerModel(TransformerModel):
    @torch.no_grad()
    def forward(self, sample_rss, sample_pos_encoding, query_pos_encoding):
        sample_rss_embedding = self.rss_bembedding(sample_rss.unsqueeze(-1))
        encoder_input = sample_rss_embedding + sample_pos_encoding
        decoder_input = query_pos_encoding

        if self.perceiver_io_flag:
            output = self.transformer(encoder_input, queries=decoder_input)
        else:
            output = self.transformer(
                src=encoder_input, 
                tgt=decoder_input
            )

        output = self.output_layer(output).squeeze(-1)
        return utils.scale_in_last_dim(output)

    @torch.no_grad()
    def prepare_positional_encoding(self, sample_pos_enc, query_pos_enc):
 
        sample_pos_encoding = self.positional_encoding(sample_pos_enc)
        query_pos_encoding = self.positional_encoding(query_pos_enc)
        return sample_pos_encoding, query_pos_encoding

def Transformer_base_config(dim, array_factor_len, num_encoder_layers, num_decoder_layers, nhead,
                             estimator_pretrained_model=None, estimator_lr_scheduler_type="cosine"):
    config = SimpleNamespace(
        type="standard",
        dim=dim,
        dim_feedforward=dim * 2,
        array_factor_len=array_factor_len,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        nhead=nhead,
        estimator_pretrained_model=estimator_pretrained_model,
        estimator_lr_scheduler_type=estimator_lr_scheduler_type,
        optimizer="adamw",
    )
    return config

def benchmark_single_model(model, model_name='Model', device='cuda:0', num_runs=100):

    sample_numbers = [64]
    query_numbers = [1600 // 4, 1600 // 2, 1600, 1600*2, 1600 * 4, 1600 * 8,1600 * 16]
    
    results = []
    
    with torch.no_grad():
        for sample_num in sample_numbers:
            for query_num in query_numbers:
                print(f"\nTesting {model_name}: sample_number={sample_num}, query_number={query_num}")
                
                sample_rss = torch.randn(1, sample_num).to(device)
                sample_pos_enc = torch.randn(1, sample_num, 1024).to(device)
                query_pos_enc = torch.randn(1, query_num, 1024).to(device)
                
                sample_pos_encoding, query_pos_encoding = model.prepare_positional_encoding(
                    sample_pos_enc, query_pos_enc
                )
                
                for _ in range(3):
                    _ = model(sample_rss, sample_pos_encoding, query_pos_encoding)
                torch.cuda.synchronize() 
                
                reset_peak_memory_stats(device)
                _ = model(sample_rss, sample_pos_encoding, query_pos_encoding)
                torch.cuda.synchronize()
                memory_mb = max_memory_allocated(device) / (1024 ** 2)
                
                times = []
                for i in range(num_runs):
                    torch.cuda.synchronize() 
                    start_time = time.perf_counter()
                    
                    _ = model(sample_rss, sample_pos_encoding, query_pos_encoding)
                    
                    torch.cuda.synchronize()  
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                    
                    if (i + 1) % 20 == 0:
                        print(f"  Progress: {i+1}/{num_runs} runs completed")
                
                time_mean = np.mean(times) * 1000  # ms
                time_std = np.std(times) * 1000
                
                results.append({
                    'model': model_name,
                    'sample_number': sample_num,
                    'query_number': query_num,
                    'time_mean_ms': round(time_mean, 3),
                    'time_std_ms': round(time_std, 3),
                    'memory_mb': round(memory_mb, 2)
                })
                
                print(f"  Results: Time={time_mean:.3f}±{time_std:.3f}ms, Memory={memory_mb:.2f}MB")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # mini model was tested
    layer_number = 2
    dim = 256
    nhead= 8
    num_latents = 16

    PerceiverIO_config_ins = PerceiverIO_config(depth = layer_number, array_factor_len = 1024, dim = dim, num_latents = num_latents, latent_dim = dim, cross_heads = nhead, latent_heads = nhead)
    pi_model = FastTransformerModel(PerceiverIO_config_ins)
    pi_model.eval()
    pi_model.to('cuda:0')
    utils.count_parameters(pi_model, name="PerceiverIO")

    print("Starting benchmark...")
    results_df = benchmark_single_model(pi_model, device='cuda:0', num_runs=1000)

    print("\n" + "="*80)
    print("Benchmark Results:")
    print("="*80)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(_OUTPUT_DIR, 'model_benchmark_results_pi.csv'), index=False)

    Transformer_base_config_ins = Transformer_base_config(dim = dim, array_factor_len = 1024, num_encoder_layers = layer_number *4, num_decoder_layers = layer_number *4, nhead = nhead)
    transformer_model = FastTransformerModel(Transformer_base_config_ins)
    transformer_model.eval()
    transformer_model.to('cuda:0')

    utils.count_parameters(transformer_model, name="Transformer")

    print("Starting benchmark...")
    results_df = benchmark_single_model(transformer_model, device='cuda:0', num_runs=100)

    print("\n" + "="*80)
    print("Benchmark Results:")
    print("="*80)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(_OUTPUT_DIR, 'model_benchmark_results_transformer.csv'), index=False)




