#!/bin/bash

CUR_DIR=$(pwd)

docker run --gpus all --rm \
    --shm-size 16G \
    -v ${CUR_DIR}/ae_figures:/app/figures \
    -v ${CUR_DIR}/csi-dataset:/app/csi-dataset \
    -v ${CUR_DIR}/ARN_saved_models:/app/ARN_saved_models \
    -v ${CUR_DIR}/saved_models:/app/saved_models \
    beamformer-docker