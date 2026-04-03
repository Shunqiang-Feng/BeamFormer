"""
inference_helper.py — Shared inference pipeline for visualize_spectrum and cdf_plot.

The model's core interface is RSS (Received Signal Strength), not raw CSI.
- visualize_spectrum.py:     computes RSS from CSI files, then calls this helper
- cdf_plot.py:               computes RSS from CSI files, then calls this helper
"""

import warnings
import torch

from beamformer.dataset import load_data_process
from beamformer.modules import AmplitudeRecoveryNetwork, FastTransformerModel
from beamformer.weight_generator import ParametricGenerator, transform_weights
from beamformer.utils import count_parameters


class InferenceHelper:
    def __init__(self, setting):
        """
        Args:
            setting: a model SimpleNamespace containing
                     .estimator, .generator, .arn_model, .assumption, .dataset
        """
        self.setting = setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dp = load_data_process(setting, device=self.device)
        self.generator = self._load_generator()
        self.estimator = self._load_estimator()
        self.arn_model = self._load_arn()

    def _load_generator(self):
        setting = self.setting
        ds = setting.dataset
        generator = ParametricGenerator(setting.assumption.sample_num, ds.M, ds.N)
        if setting.generator.generator_pretrained_model:
            generator.load_state_dict(
                torch.load(setting.generator.generator_pretrained_model,
                           map_location=self.device, weights_only=True)
            )
            print(f"Loaded generator from {setting.generator.generator_pretrained_model}")
        generator.eval()
        generator.to(self.device)
        count_parameters(generator, "Generator")
        return generator

    def _load_estimator(self):
        cfg = self.setting.estimator
        estimator = FastTransformerModel(cfg)
        estimator.load_state_dict(
            torch.load(cfg.estimator_pretrained_model,
                       map_location=self.device, weights_only=True)
        )
        print(f"Loaded estimator from {cfg.estimator_pretrained_model}")
        estimator.to(self.device)
        count_parameters(estimator, "Estimator")
        return estimator

    def _load_arn(self):
        cfg = self.setting.arn_model
        sample_num = self.setting.assumption.sample_num
        arn_model = AmplitudeRecoveryNetwork(sample_num=sample_num, d_model=cfg.hidden_dims)
        arn_model.has_pretrained = False
        if cfg.ARN_model_pretrained_model is not None:
            arn_model.load_state_dict(
                torch.load(cfg.ARN_model_pretrained_model,
                           map_location=self.device, weights_only=True)
            )
            arn_model.has_pretrained = True
            print(f"Loaded ARN from {cfg.ARN_model_pretrained_model}")
        else:
            warnings.warn("No ARN pretrained model — amplitude will be set to 1.", RuntimeWarning)
        arn_model.eval()
        arn_model.to("cpu")
        return arn_model

    @torch.no_grad()
    def infer_from_csi(self, csi_tensor):
        """
        Full pipeline: CSI → generate weights → compute RSS → estimator.

        Args:
            csi_tensor: [1, freq, rx, tx] complex tensor

        Returns:
            sample_rss:      [sample_num]  normalized RSS values used as model input
            query_rss:       [angle_spectrum_length]  ground-truth angle spectrum (normalized)
            query_rss_pred:  [angle_spectrum_length]  predicted angle spectrum
            scale:           scalar, the normalization factor
            weights:         [1, sample_num, M, N] beam weights used
        """
        csi = csi_tensor.to(self.device)
        batch_size = csi.shape[0]

        z = torch.randn(batch_size, self.setting.assumption.sample_num *
                        self.setting.dataset.M * self.setting.dataset.N).to(self.device)
        raw_weights = self.generator(z)
        weights, _ = transform_weights(raw_weights)

        sample_rss, scale, query_rss, query_rss_pred = self._run_estimator(weights, csi)
        return (
            sample_rss[0].cpu(),
            query_rss[0].cpu(),
            query_rss_pred[0].cpu(),
            scale[0].item(),
            weights.cpu(),
        )

    @torch.no_grad()
    def infer_from_rss(self, sample_rss_input, weights):
        """
        Partial pipeline: pre-computed RSS → estimator (skips CSI loading).

        Args:
            sample_rss_input: [sample_num] tensor, already normalized (max = 1)
            weights:          [1, sample_num, M, N] beam weights (for positional encoding)

        Returns:
            query_rss_pred: [angle_spectrum_length] predicted angle spectrum
        """
        sample_rss = sample_rss_input.unsqueeze(0).to(device=self.device, dtype=torch.float32)
        weights = weights.to(self.device)

        sample_pos_enc = self.dp.generate_sample_position_encoding(weights).to(dtype=torch.float32)
        query_pos_enc = self.dp.generate_query_position_encoding(batch_size=1).to(dtype=torch.float32)
        sample_pos_encoding, query_pos_encoding = self.estimator.prepare_positional_encoding(
            sample_pos_enc, query_pos_enc
        )
        query_rss_pred, _ = self.estimator(sample_rss, sample_pos_encoding, query_pos_encoding)
        return query_rss_pred[0].cpu()

    def apply_arn(self, sample_rss, query_rss_pred, scale):
        """
        Apply ARN amplitude correction to the predicted spectrum.

        Args:
            sample_rss:      [sample_num] normalized RSS tensor (cpu)
            query_rss_pred:  [angle_spectrum_length] predicted spectrum (cpu)
            scale:           scalar normalization factor

        Returns:
            pred_angle_spectrum: [80, 20] corrected angle spectrum (numpy)
        """
        from beamformer.utils import gpu_tensor_to_np

        if self.arn_model.has_pretrained:
            pred_amp_inv = self.arn_model(sample_rss.unsqueeze(0))
            pred_lobe = 1.0 / torch.mean(pred_amp_inv).item()
        else:
            pred_lobe = 1.0

        pred_spectrum = query_rss_pred.reshape(80, 20) * pred_lobe * scale
        return gpu_tensor_to_np(pred_spectrum)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_estimator(self, weights, csi):
        """Core estimator forward pass given weights and csi (both on device)."""
        sample_pos_enc = self.dp.generate_sample_position_encoding(weights).to(dtype=torch.float32)
        query_pos_enc = self.dp.generate_query_position_encoding(
            batch_size=csi.shape[0]
        ).to(dtype=torch.float32)

        sample_rss = self.dp.generate_sample_rss(csi, weights).to(dtype=torch.float32)
        query_rss = self.dp.generate_query_rss(csi).to(dtype=torch.float32)

        scale = torch.max(sample_rss, dim=1, keepdim=True).values
        sample_rss = sample_rss / scale
        query_rss = query_rss / scale

        sample_pos_encoding, query_pos_encoding = self.estimator.prepare_positional_encoding(
            sample_pos_enc, query_pos_enc
        )
        query_rss_pred, _ = self.estimator(sample_rss, sample_pos_encoding, query_pos_encoding)

        return sample_rss, scale, query_rss, query_rss_pred
