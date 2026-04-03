#!/bin/bash

CUR_DIR=$(pwd)

docker run --gpus all --rm \
    --shm-size 16G \
    -v ${CUR_DIR}/figures:/app/figures \
    -v ${CUR_DIR}/mini_demo:/app/mini_demo \
    -v ${CUR_DIR}/baselines_result:/app/baselines_result \
    beamformer-docker