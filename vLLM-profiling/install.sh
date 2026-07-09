#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
NSIGHT_SYS_VERSION=2026.3.1.157-3804839
CAPPED_NSIGHT_SYS_VERSION=$(echo ${NSIGHT_SYS_VERSION} | cut -d. -f1-3)
NSIGHT_SYSTEMS_CLI_PATH=/opt/nvidia/nsight-systems-cli

# create python virtual env
rm -rf ${THIS_DIR}/.venv
python3 -m venv ${THIS_DIR}/.venv

# vLLM (pulls a matching torch/cuda build)
${THIS_DIR}/.venv/bin/python -m pip install --upgrade pip
${THIS_DIR}/.venv/bin/python -m pip install vllm huggingface_hub

# Confirm the driver itself is healthy before touching packages
nvidia-smi || { echo "GPU/driver not responding — stop here, don't proceed"; exit 1; }

wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_3/NsightSystems-linux-cli-public-${NSIGHT_SYS_VERSION}.deb -O ${THIS_DIR}/nsight-systems.deb
sudo apt install ${THIS_DIR}/nsight-systems.deb

# # What did dpkg actually complain about? (rerun to see the real error, not swallowed by the script)
sudo dpkg -i ${THIS_DIR}/nsight-systems.deb

# # If that reports missing deps, resolve them narrowly:
sudo apt-get install -f -y --no-install-recommends

${NSIGHT_SYSTEMS_CLI_PATH}/${CAPPED_NSIGHT_SYS_VERSION}/bin/nsys --version


