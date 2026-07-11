#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")

# create python virtual env
rm -rf ${THIS_DIR}/.venv
python3 -m venv ${THIS_DIR}/.venv

# vLLM (pulls a matching torch/cuda build)
${THIS_DIR}/.venv/bin/python -m pip install --upgrade pip
${THIS_DIR}/.venv/bin/python -m pip install vllm huggingface_hub

# Confirm the driver itself is healthy before touching packages
nvidia-smi || { echo "GPU/driver not responding — stop here, don't proceed"; exit 1; }

# confirm nsys is installed and working
nsys --version || { echo "nsys not found or not working — stop here, don't proceed"; exit 1; }
