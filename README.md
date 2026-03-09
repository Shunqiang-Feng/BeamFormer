# BeamFormer — Artifact Evaluation

This repository contains the artifact for reproducing the experimental results and figures presented in the paper. Follow the steps below to set up the environment and reproduce all results.

---

## Requirements

**Hardware**

| Component | Requirement |
|---|---|
| GPU VRAM | >= 20 GB |
| Disk Space | ~34 GB (for downloaded archives and extracted data) |

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
| `prepare_files.sh` (download ~33.6 GiB) | Depends on network bandwidth |
| Full artifact reproduction (`from_scratch`) | ~2–3 hours |
| 2ACE baseline from scratch (compressed sensing) | > 12 hours (see note below) |

---

## Quick Start

### Step 1 — Download Datasets and Model Checkpoints

```bash
bash prepare_files.sh
```

This script downloads and extracts the following archives from the artifact repository:

| Archive | Contents |
|---|---|
| `csi-dataset.tar.gz` | CSI measurement dataset |
| `saved_models.tar.gz` | Pre-trained BeamFormer model weights |
| `ARN_saved_models.tar.gz` | Pre-trained ARN baseline model weights |

### Step 2 — Build the Docker Image

```bash
docker build -t beamformer-docker .
```

The image is based on `huggingface/accelerate:gpu-nightly` and installs all required Python dependencies listed in [requirements.txt](requirements.txt).

### Step 3 — Run the Artifact

```bash
bash run_docker.sh
```

The container mounts the local dataset and model directories into `/app` and internally executes `run_reproduce_figures.sh --data_source from_scratch`, which reproduces **all figures from the paper** in a single run. Upon completion, all generated figures are saved to `./ae_figures/` on the host machine. No further steps are required for the main evaluation.

---

## Reproducing Figures

Inside the container (or in a local environment), figures can be reproduced individually via:

```bash
python reproduce_figures.py --figure "<figure_name>" --data_source [cache|from_scratch]
```

The following figures are supported:

| Figure | Description |
|---|---|
| Overall performance of different approaches | Figure 10 |
| Visualization of layer-wise beam spectrum generation | Figure 11 |
| Impact of scene configurations | Figure 13 |
| Impact of model parameters | Figure 14 |
| Comparison of positional encoders | Figure 15 |
| Comparison of model latencies | Figure 16 |
| Comparison of reference beam settings | Figure 17 |
| Comparison of power estimators | Figure 18 |
| Performance with real-world data | Figure 19 |
| Example spectrum with Sivers SDR | Figure 20 |
| Example spectrum with IBM SDR | Figure 20 |

To reproduce all figures at once, use:

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

1. Set up a local Python environment (see [requirements.txt](requirements.txt); **remove all `#` comment prefixes before installing**).
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
