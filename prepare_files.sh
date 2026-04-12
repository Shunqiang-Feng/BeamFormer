#!/bin/bash

# 1. Download Files
echo "==================== Downloading Files ===================="
wget -c https://s3-west.nrp-nautilus.io/BeamFormer/ae_necessary.zip

# 2. Extract Files
echo "==================== Extracting Files ===================="
unzip -o ae_necessary.zip

# 3. Move to project directory
echo "==================== Moving Files ===================="
mv ae_necessary/ARN_saved_models .
mv ae_necessary/csi-dataset .
mv ae_necessary/saved_models .

# 4. Cleanup
rm -rf ae_necessary ae_necessary.zip

echo "==================== Setup Complete ===================="
ls -d */ | grep -E 'csi-dataset|saved_models|ARN_saved_models'