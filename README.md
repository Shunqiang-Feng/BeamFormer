# BeamFormer: Transformer-based Beam Management for 6G Networks

> **BeamFormer** is a transformer-based framework for beam management in gMIMO 6G networks. It reconstructs the full beam spectrum (RSS across all beam directions) from a small number of reference beam measurements, achieving real-time beam alignment with 1.8 ms inference latency for 1,600 beams.

Paper: *BeamFormer: Transformer-based Beam Management for 6G Networks*, MobiSys 2026 (link TBD)

---

## Quick Start

### 1. Set up the environment

```bash
conda create -n beamformer python=3.12 -y
conda activate beamformer
pip install -r requirements.txt
```

### 2. Download the dataset and pre-trained models

```bash
bash prepare_files.sh
```

This downloads and extracts `mini_demo.zip` into `./mini_demo/`, which contains example CSI data and pre-trained model weights.

### 3. Visualize beam spectra

```bash
python -m beamformer.visualize_spectrum
```

Output: `figures/spectrum_grid.png`

---

## Results

### Angle Spectrum Visualization

Ground truth vs. predicted beam spectra across 32 random CSI samples (polar disk plots, sin-projection):

```bash
python -m beamformer.visualize_spectrum
```

![Spectrum Grid](figures/spectrum_grid.png)

### CDF Comparison

BeamFormer vs. baselines (AgileLink, Coarse/Fine Sweep, Hier. Sweep, 2ACE, MLP, CNN) on the full test set:

```bash
python -m beamformer.cdf_plot
```

![CDF Comparison](figures/cdf_comparison.png)

---

## Dataset

The full training dataset is publicly available:

- **IEEE DataPort**: [10.21227/g1zj-z323](https://ieee-dataport.org/documents/beamformer-ray-tracing-channel-dataset)
- **S3**: [Nautilus Ceph Storage](https://s3-west.nrp-nautilus.io/BeamFormer/dataset/homeoffice-communication-28G-raw-data.zip)
- **Data processing code**: [github.com/Shunqiang-Feng/BeamFormer-Dataset](https://github.com/Shunqiang-Feng/BeamFormer-Dataset)

---

## Training Your Own Model

### Prepare the Dataset

First, download the open-source raw dataset (see the [Dataset](#dataset) section above). This is a raw channel database, not ready-to-use CSI. Next, use the data processing code at [github.com/Shunqiang-Feng/BeamFormer-Dataset](https://github.com/Shunqiang-Feng/BeamFormer-Dataset) to generate the CSI dataset: edit `config.m` to match your antenna configuration and operating frequency, then run `main.m`.

**A note on frequency generalization:** Although the dataset is simulated at 28 GHz, the trained model transfers directly to other frequencies — no retraining required. The key is to ensure the normalized antenna spacing is consistent between simulation and deployment, i.e.:

$$\frac{d_\text{sim}}{\lambda_\text{sim}} = \frac{d_\text{real}}{\lambda_\text{real}}$$

where $d_\text{real}$ and $\lambda_\text{real}$ are the physical antenna spacing and wavelength of your real array, and $d_\text{sim}$ and $\lambda_\text{sim}$ are the corresponding values used in the simulation (at 27.925 GHz). To satisfy this, set the simulation spacing to:

$$d_\text{sim} = d_\text{real} \cdot \frac{\lambda_\text{sim}}{\lambda_\text{real}}$$

### Configuration

Edit the config files in [`configs/`](configs/) to match your hardware and dataset paths:

| File | Purpose |
|---|---|
| [`configs/co_train.py`](configs/co_train.py) | Beam Generator + Latent Beam Processor |
| [`configs/arn.py`](configs/arn.py) |  Beam Power Estimator (ARN) |
| [`configs/submodules.py`](configs/submodules.py) | Shared component definitions (model architecture, dataset) |

Key parameters to adjust in `configs/submodules.py`:
- `dataset.homeoffice_communication_28g()` — update `train_data_path` and `test_data_path`
- `estimator.PerceiverIO(depth=..., dim=...)` — model capacity
- `training.co_train(...)` — batch size, epochs, learning rates, GPU count

### Stage 1 & 2 — Beam Generator + Latent Beam Processor (co-training)

```bash
accelerate launch -m beamformer.train --config co_train
```

Model weights are saved to `saved_models/co_train/`.

### Stage 3 — Beam Power Estimator (ARN)

Update `generator_pretrained_model` in `configs/arn.py` to point to the Stage 2 generator weights, then:

```bash
accelerate launch -m beamformer.train_ARN --config arn
```

Model weights are saved to `ARN_saved_models/arn/`.

---

## Deployment

At inference time, the base station:

1. Uses the trained Beam Generator to generate reference beam weight.
2. Measures RSS for each reference beam.
3. Feeds the RSS into BeamFormer to reconstruct the full beam spectrum.
4. Selects the beam direction with the highest predicted RSS for alignment.

