from types import SimpleNamespace
import os
import copy
from scipy.constants import c as light_speed


class assumption:
    @staticmethod
    def beam64():
        """Reference beam sampling and beam spectrum resolution settings."""
        config = SimpleNamespace(
            sample_num=64,                   # number of reference beams swept at the base station
            angle_steps_theta=20,            # elevation bins of the output beam spectrum (θ ∈ [0°, 90°])
            angle_steps_phi=80,              # azimuth bins of the output beam spectrum (φ ∈ [0°, 360°))
            array_factor_steps_theta=16,     # elevation bins for array-factor (beam pattern) encoding
            array_factor_steps_phi=64,       # azimuth bins for array-factor encoding → 1024-dim AF vector
        )
        config.angle_spectrum_length = config.angle_steps_theta * config.angle_steps_phi  # total query beams (1600)
        return config


class dataset:
    @staticmethod
    def homeoffice_communication_28g(add_noise=False, snr_min=None):
        """28 GHz indoor home-office dataset with a 16×16 Tx URA and 2×1 Rx array."""
        DATA_FOLDER = "./csi-dataset"
        ds = SimpleNamespace(
            name='homeoffice_communication_28g',
            train_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t16x16_r2x1_train",
            test_data_path=f"{DATA_FOLDER}/homeoffice-communication-28G-csi/t16x16_r2x1_test_small",
            mode="rx_act1",          # steer Tx array; Rx uses a single activated element (quasi-omni)
            freq_num=128,            # OFDM sub-carriers; RSS is averaged across sub-carriers to suppress fast fading
            start_freq=27.90964e9,
            end_freq=27.94012e9,
            M_tx=16,                 # Tx URA rows (256 elements total)
            N_tx=16,                 # Tx URA columns
            M_rx=2,                  # Rx array rows
            N_rx=1,                  # Rx array columns
            max_theta=90,            # maximum elevation angle of the beam spectrum
            add_noise=add_noise,     # whether to add simulated thermal noise to CSI
            snr_min=snr_min,         # SNR floor (dB) when add_noise=True
        )
        if ds.mode == "rx_act1":
            ds.M = ds.M_tx
            ds.N = ds.N_tx
        else:
            ds.M = ds.M_rx
            ds.N = ds.N_rx
        mid_freq = 27.925e9
        ds.d_row = 0.5 * light_speed / mid_freq   # half-wavelength row spacing (m)
        ds.d_col = 0.5 * light_speed / mid_freq   # half-wavelength column spacing (m)
        ds.subcarrier_spacing = (ds.end_freq - ds.start_freq) / (ds.freq_num - 1)  # Δf for noise power computation
        return ds


class estimator:
    @staticmethod
    def PerceiverIO(depth=8, array_factor_len=1024, dim=1024, num_latents=64, latent_dim=1024,
                    cross_heads=8, latent_heads=8, decoder_ff=True, optimizer="adamw",
                    estimator_pretrained_model=None, seq_dropout_prob=0):
        """Latent Beam Processor: Perceiver IO transformer that maps reference beam RSS to full beam spectrum."""
        config = SimpleNamespace(
            type="perceiver_io",
            depth=depth,                     # number of latent self-attention layers
            array_factor_len=array_factor_len, # input token dimension = AF encoding size (16×64=1024)
            dim=dim,                         # projection dimension of input tokens
            num_latents=num_latents,         # number of learnable latent tokens (compact channel representation)
            latent_dim=latent_dim,           # feature dimension of each latent token
            cross_heads=cross_heads,         # attention heads for cross-attention (context→latent, latent→query)
            latent_heads=latent_heads,       # attention heads for latent self-attention
            decoder_ff=decoder_ff,           # whether to add FFN after decoder cross-attention
            estimator_pretrained_model=estimator_pretrained_model,
            optimizer=optimizer,
            seq_dropout_prob=seq_dropout_prob,  # input token dropout for robustness to missing reference beams
        )
        config.dim_feedforward = config.array_factor_len * 2  # FFN hidden size (2× expansion)
        config.queries_dim = config.dim
        config.cross_dim_head = config.queries_dim // config.cross_heads
        config.latent_dim_head = config.latent_dim // config.latent_heads
        return config


class generator:
    @staticmethod
    def parametric_generator(generator_pretrained_model=None):
        """Beam Generator: learnable module that optimizes reference beam weight vectors end-to-end."""
        return SimpleNamespace(
            name='PARAM',
            type='PARAM',
            generator_pretrained_model=generator_pretrained_model,
            phase_constraint=False,   # phase quantization constraint for hardware-limited phased arrays
        )


class training:
    @staticmethod
    def co_train(local_config_name):
        """Co-training config for the Beam Pattern Encoder, Latent Beam Processor, and Beam Generator."""
        return SimpleNamespace(
            batch_size=120,          # per-GPU batch size (effective batch = 120 × 4 GPUs = 480)
            warmup_ratio=0.05,       # fraction of steps for linear LR warm-up
            learning_rate=1e-5,      # default LR (overridden per module below)
            lr_generator=1e-4,       # higher LR for the Beam Generator (trained from scratch in stage 2)
            lr_estimator=1e-5,       # LR for the Beam Pattern Encoder + Latent Beam Processor
            random_seed=42,
            model_save_path=os.path.join("saved_models", local_config_name),
            epochs=30,               # co-training epochs (stage 2 of the three-stage training scheme)
            scheme='co-train',       # selects the joint generator+estimator optimization code path
            num_workers=0,
            gpu_num=4,               # number of GPUs for distributed training via HuggingFace Accelerate
        )


class ARN_model:
    @staticmethod
    def typical_ARN(ARN_model_pretrained_model=None):
        """Beam Power Estimator (ARN): predicts the scale ratio α to recover absolute RSS from normalized beam spectra."""
        return SimpleNamespace(
            output_num=16,           # number of scale-ratio predictions averaged for robustness
            hidden_dims=512,         # feature dimension of the ARN transformer encoder
            ARN_model_pretrained_model=ARN_model_pretrained_model,
        )


class ARN_training:
    @staticmethod
    def add_train_ARN(config_original, config_name=None):
        """Attach stage-3 training config for the ARN; generator and estimator weights are frozen."""
        config = copy.deepcopy(config_original)
        if config_name is None:
            raise ValueError("You need input config name")
        config.training = SimpleNamespace(
            batch_size=400,          # larger batch than co-training; ARN is lightweight and converges quickly
            warmup_ratio=0.05,
            learning_rate=1e-5,
            random_seed=42,
            model_save_path=os.path.join("ARN_saved_models", config_name),
            epochs=10,               # stage-3 training epochs
            num_workers=0,
            gpu_num=4,
        )
        return config
