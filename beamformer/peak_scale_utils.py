#!/usr/bin/env python3


import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
from .utils import scale_in_last_dim, load_config
from .dataset import load_datasets,load_data_process
from .weight_generator import transform_weights


def compute_peak_scale_statistics(
    config,
    num_samples: int = 10000,
    batch_size: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True
) -> Tuple[float, float]:
 
    if batch_size is None:
        batch_size = config.training.batch_size

    train_dataset, _ = load_datasets(config)
    print(f"The length of dataset is: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    from train_ARN import Trainer
    trainer = Trainer(config)
    generator = trainer.initialize_generator().to(device)

    if trainer.is_generator_ml():
        generator.eval()

    dp = load_data_process(config, device=device)

    peak_scales = []
    samples_collected = 0

    iterator = tqdm(train_dataloader, desc="Computing peak_scale statistics") if verbose else train_dataloader

    with torch.no_grad():
        for csi, _ in iterator:
            if samples_collected >= num_samples:
                break

            csi = csi.to(device)
            current_batch_size = csi.shape[0]

            if trainer.is_generator_ml():
                z_dim = config.assumption.sample_num * config.dataset.M * config.dataset.N
                z = torch.randn(current_batch_size, z_dim).to(device)
                raw_weights = generator(z)
                weights_out, _ = transform_weights(raw_weights)
            else:
                weights_out, _ = generator.generate(batch_size=current_batch_size)

            sample_rss = dp.generate_sample_rss(csi, weights_out)
            query_rss = dp.generate_query_rss(csi)

            scale = torch.max(sample_rss, dim=1, keepdim=True).values
            sample_rss /= scale
            query_rss /= scale

            _, peak_scale = scale_in_last_dim(query_rss)

            peak_scales.append(peak_scale.cpu())
            samples_collected += current_batch_size

            if verbose:
                iterator.set_postfix({
                    'samples': f'{samples_collected}/{num_samples}'
                })

    peak_scales = torch.cat(peak_scales, dim=0)[:num_samples]

    mean = peak_scales.mean().item()
    std = peak_scales.std().item()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Peak Scale Statistics (n={num_samples}):")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std:  {std:.6f}")
        print(f"  Min:  {peak_scales.min().item():.6f}")
        print(f"  Max:  {peak_scales.max().item():.6f}")
        print(f"{'='*60}")

    return mean, std


class PeakScaleGenerator:

    def __init__(
        self,
        mean: float,
        std: float,
        distribution: str = "normal",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):

        self.mean = mean
        self.std = std
        self.distribution = distribution.lower()
        self.device = device

        if self.distribution not in ["normal", "lognormal"]:
            raise ValueError(f"Unsupported distribution: {distribution}. Use 'normal' or 'lognormal'.")

        if self.distribution == "lognormal":
            # mu = log(mean^2 / sqrt(mean^2 + std^2))
            # sigma = sqrt(log(1 + std^2 / mean^2))
            self.log_mean = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            self.log_std = np.sqrt(np.log(1 + std**2 / mean**2))

    def generate(
        self,
        batch_size: int,
        sample_num: int,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ) -> torch.Tensor:

        shape = (batch_size, sample_num)

        if self.distribution == "normal":
            peak_scales = torch.randn(shape, device=self.device) * self.std + self.mean
        else:  # lognormal
            log_values = torch.randn(shape, device=self.device) * self.log_std + self.log_mean
            peak_scales = torch.exp(log_values)

        if clip_min is not None:
            peak_scales = torch.clamp(peak_scales, min=clip_min)
        if clip_max is not None:
            peak_scales = torch.clamp(peak_scales, max=clip_max)

        return peak_scales

    def __repr__(self) -> str:
        return (
            f"PeakScaleGenerator(mean={self.mean:.4f}, std={self.std:.4f}, "
            f"distribution='{self.distribution}', device='{self.device}')"
        )


def save_peak_scale_statistics(mean: float, std: float, save_path: str):

    torch.save({
        'mean': mean,
        'std': std,
    }, save_path)
    print(f"Peak scale statistics saved to: {save_path}")


def load_peak_scale_statistics(load_path: str) -> Tuple[float, float]:

    stats = torch.load(load_path, weights_only=True)
    mean = stats['mean']
    std = stats['std']
    print(f"Loaded peak scale statistics: mean={mean:.6f}, std={std:.6f}")
    return mean, std


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Peak Scale Statistics and Generation")
    parser.add_argument("--config", type=str, required=True, help="Config name (e.g., 'pi_param-64-indoor')")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for statistics")
    parser.add_argument("--save", type=str, default=None, help="Path to save statistics (optional)")
    parser.add_argument("--test_generate", action="store_true", help="Test generation after computing statistics")

    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config, predix="train_ARN")

    print("\n[Step 1] Computing peak_scale statistics...")
    mean, std = compute_peak_scale_statistics(
        config,
        num_samples=args.num_samples,
        verbose=True
    )

    if args.save:
        save_peak_scale_statistics(mean, std, args.save)

    if args.test_generate:
        print("\n[Step 2] Testing PeakScaleGenerator...")
        generator = PeakScaleGenerator(mean, std)
        print(generator)

        test_peak_scales = generator.generate(batch_size=32, sample_num=64)
        print(f"\nGenerated peak_scales shape: {test_peak_scales.shape}")
        print(f"Generated statistics:")
        print(f"  Mean: {test_peak_scales.mean().item():.6f}")
        print(f"  Std:  {test_peak_scales.std().item():.6f}")
        print(f"  Min:  {test_peak_scales.min().item():.6f}")
        print(f"  Max:  {test_peak_scales.max().item():.6f}")
