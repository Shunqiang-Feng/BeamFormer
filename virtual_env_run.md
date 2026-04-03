# BeamFormer Environment Setup and Reproduction Guide

## 1. Download and Extract Dataset & Checkpoints
```bash
./prepare_files.sh
```
This script downloads the dataset and model checkpoints from remote storage, then extracts them.

## 2. Create Conda Environment
```bash
conda create -n accelerate python=3.12 -y
conda activate accelerate
```

## 3. Update requirements.txt

```bash
cat > requirements.txt << 'EOF'
accelerate==1.12.0
einops==0.8.2
matplotlib==3.10.8
numpy==2.4.2
pandas==3.0.1
PyYAML==6.0.3
rich==14.3.3
scipy>=1.10.1
scikit-image>=0.19.0
torch==2.9.1
tqdm==4.67.1
perceiver_pytorch
EOF
```

## 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 5. Run Reproduction Script
```bash
./run_reproduce_figures.sh --data_source from_scratch
```