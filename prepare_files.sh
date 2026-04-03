#!/bin/bash
set -e

echo "[prepare] Downloading mini_demo.zip ..."
wget -q --show-progress https://s3-west.nrp-nautilus.io/BeamFormer/mini_demo.zip

echo "[prepare] Extracting ..."
unzip -q mini_demo.zip
rm mini_demo.zip

echo "[prepare] Done. Contents extracted to ./mini_demo/"
