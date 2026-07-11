#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
NSIGHT_SYS_VERSION=2026.3.1.157-3804839
CAPPED_NSIGHT_SYS_VERSION=$(echo ${NSIGHT_SYS_VERSION} | cut -d. -f1-3)
NSIGHT_SYSTEMS_CLI_PATH=/opt/nvidia/nsight-systems-cli

source ${THIS_DIR}/.env
source ${THIS_DIR}/.venv/bin/activate

python ${THIS_DIR}/bench.py

# # now the real profiled run
# ${NSIGHT_SYSTEMS_CLI_PATH}/${CAPPED_NSIGHT_SYS_VERSION}/bin/nsys profile \
#   --trace=cuda,nvtx,osrt \
#   --cuda-graph-trace=node \
#   --output=vllm_llama3_8b_bs32 \
#   --force-overwrite=true \
#   python ${THIS_DIR}/bench.py


nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --output=vllm_llama3_8b_bs32 \
  --force-overwrite=true \
  python ${THIS_DIR}/bench.py