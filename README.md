# BeamFormer — Artifact Evaluation

This repository contains the artifact for reproducing the experimental results and figures presented in the paper. Follow the steps below to set up the environment and reproduce all results.

> **Important:** Always use the latest code from the [`AE` branch](https://github.com/Shunqiang-Feng/BeamFormer/tree/AE) of this repository to reproduce experiments. The DOI archived version has not undergone extensive testing and may be outdated.

---

## Requirements

**Hardware**

| Component | Requirement |
|---|---|
| GPU VRAM | >= 20 GB |
| Disk Space | ~56 GB (for downloaded archive and extracted data) |

**Tested Configurations**

The artifact has been verified on the following GPU configurations:

| GPU | VRAM | NVIDIA Driver | CUDA (Host) |
|---|---|---|---|
| NVIDIA RTX 4000 Ada Generation | 20 GB | 573.44 | 12.8 |
| NVIDIA A40 | 46 GB | 580.95.05 | 13.0 |

Other driver/CUDA versions or GPU models may work but have not been tested.

**Software**

| Component | Requirement |
|---|---|
| Docker | with `--gpus all` support |


**Estimated Runtimes**

| Task | Estimated Time |
|---|---|
| `prepare_files.sh` (download ~56 GB) | Depends on network bandwidth |
| Full artifact reproduction (`from_scratch`) | ~2–3 hours |
| 2ACE baseline from scratch (compressed sensing) | > 12 hours (see note below) |

---

## Quick Start

### Step 1 — Build the Docker Image

```bash
docker build -t beamformer-docker .
```

The image is based on `huggingface/accelerate:gpu-nightly` and installs all required Python dependencies listed in [requirements.txt](requirements.txt).

> **Note:** Build the Docker image **before** running `prepare_files.sh`. The large dataset and model files downloaded by `prepare_files.sh` would otherwise be copied into the Docker build context, significantly slowing down the build.

### Step 2 — Download Datasets and Model Checkpoints

```bash
bash prepare_files.sh
```

This script downloads `ae_necessary.zip` from the artifact repository, extracts it, and moves the following directories into the project root:

| Directory | Contents |
|---|---|
| `csi-dataset/` | CSI measurement dataset |
| `saved_models/` | Pre-trained model weights of **Beam Generator, Beam Pattern Encoder & Latent Beam Processor** |
| `ARN_saved_models/` | Pre-trained **Beam Power Estimator** model weights |

### Step 3 — Run the Artifact

```bash
bash run_docker.sh
```

The container mounts the local dataset and model directories into `/app` and internally executes `run_reproduce_figures.sh --data_source from_scratch`, which reproduces **Figures 11–25** in a single run. Upon completion, all generated figures are saved to `./ae_figures/` on the host machine. No further steps are required for the main evaluation.

---

## Reproducing Figures

Inside the container (or in a local environment), figures can be reproduced individually via:

```bash
python reproduce_figures.py --figure "<figure_name>" --data_source [cache|from_scratch]
```

Use `--list` to see all available figures:

```bash
python reproduce_figures.py --list
```

The following figures are supported:

| Figure | Title | Description |
|---|---|---|
| 10 | Overall performance of different approaches | 2×2 CDF: (RSS Loss, AoD Error) × (LoS, NLoS), indoor+outdoor combined |
| 11 | Impact of scene configurations | CDFs across frequency, scenario, array size, and SNR |
| 12 | Failure case under severe multi-paths | Beam spectrum visualization (GT & Pred) for multi-path failure case |
| 13 | Failure case under pure noise | Beam spectrum visualization (GT=zeros, Pred) for pure-noise input |
| 14 | Beam Spectrum Resolution | 2×4 polar disk: GT vs. predicted beam spectra for varying K-factor scenarios |
| 15 | Multi Path Prediction Accuracy | CDF of RSS prediction error for 1st/2nd/3rd strongest paths |
| 16 | Comparison of model latencies | Inference latency comparison across model variants |
| 17 | Impact of latency and user mobility | Mean RSS Loss vs. latency for different user speeds |
| 18 | Comparison of positional encoders | CDF comparing Array Factor, 2D Positional, and Concat position encodings |
| 19 | Comparison of reference beam settings | CDF across different beam settings |
| 20 | Comparison of power estimators | RSS error CDF comparing ARN power estimator vs. Normal Distribution |
| 21 | Impact of model parameters | Ablation study CDFs |
| 22 | Performance with real-world data | CDF from real-world Sivers/IBM hardware evaluations |
| 23 | Example spectrum with Sivers SDR | Beam spectrum visualization from Sivers SDR |
| 24 | Example spectrum with IBM SDR | Beam spectrum visualization from IBM SDR |
| 25 | UMAP Visualization of Simulation and Real-World Feature Distributions | UMAP panel: sim vs. real-world Sivers input/output beam RSS features |

To reproduce all figures at once:

```bash
bash run_reproduce_figures.sh --data_source from_scratch
```

---

## Note on Latency Results (Figure 16)

The absolute latency values reported in Figure 16 are hardware-dependent and may differ across machines. However, the **relative ordering and trends** among the compared methods are expected to remain consistent regardless of the specific device used.

---

## Note on the 2ACE Baseline (Figure 10)

The 2ACE baseline relies on MATLAB, which is proprietary software and requires a separate license. To avoid requiring reviewers to install MATLAB, pre-computed results for 2ACE are cached in `cache/2ACE/` and loaded automatically by default.

> **Warning:** 2ACE is a compressed sensing algorithm with very high computational cost. Reproducing its results from scratch is expected to take **more than 12 hours**. We strongly recommend using the pre-cached results unless full verification is required.

To reproduce the 2ACE results from scratch, follow these steps:

1. Set up a local Python environment (see [requirements.txt](requirements.txt);).
2. Install the MATLAB Engine for Python by following the official guide:
   [https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
3. Delete the cached results directory:
   ```bash
   rm -rf cache/2ACE
   ```
4. Run the figure reproduction script with `from_scratch`:
   ```bash
   python reproduce_figures.py \
       --figure "Overall performance of different approaches" \
       --data_source from_scratch
   ```
