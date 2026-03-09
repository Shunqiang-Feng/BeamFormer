#!/bin/bash

# 1. Download Files
echo "==================== Downloading Files ===================="
wget -c https://s3-west.nrp-nautilus.io/BeamFormer/dataset/csi-dataset.tar.gz
wget -c https://s3-west.nrp-nautilus.io/BeamFormer/checkpoints/ARN_saved_models.tar.gz
wget -c https://s3-west.nrp-nautilus.io/BeamFormer/checkpoints/saved_models.tar.gz

# 2. Extract Files
echo "==================== Extracting Files ===================="
tar -zxvf csi-dataset.tar.gz
tar -zxvf ARN_saved_models.tar.gz
tar -zxvf saved_models.tar.gz

# 3. Cleanup (Optional, uncomment if you want to delete .tar.gz files)
# rm *.tar.gz

echo "==================== Setup Complete ===================="
ls -d */ | grep -E 'csi-dataset|saved_models|ARN_saved_models'