#!/bin/bash
# Compile the memory-bound vs compute-bound benchmark
# Usage: ./compile.sh [sm_XX]   (default: sm_80 for A100)
#        ./compile.sh sm_86     (RTX 3090 / A10)
#        ./compile.sh sm_90     (H100)

set -e

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname "${THIS_DIR}")
ARCH="${1:-sm_80}"

echo "Building for arch=${ARCH}"

# ── Recreate build dirs ────────────────────────────────────────────────────
rm -rf "${ROOT_DIR}/build"
mkdir -p "${ROOT_DIR}/build/obj" "${ROOT_DIR}/build/bin"

INCLUDES="-I${ROOT_DIR}/include"
CUDA_INCLUDES="-I/usr/include"
CUDA_LIB_DIRS="-L/usr/lib/x86_64-linux-gnu/"
CUDA_LIB="-lcudart"
FLAGS="-O3 -arch=${ARCH} --use_fast_math"

# ── Compile objects ────────────────────────────────────────────────────────
nvcc ${FLAGS} ${INCLUDES} ${CUDA_INCLUDES} \
    -c "${ROOT_DIR}/src/kernels.cu" \
    -o "${ROOT_DIR}/build/obj/kernels.o"

nvcc ${FLAGS} ${INCLUDES} ${CUDA_INCLUDES} \
    -c "${ROOT_DIR}/src/main.cu" \
    -o "${ROOT_DIR}/build/obj/main.o"

# ── Link ───────────────────────────────────────────────────────────────────
nvcc ${FLAGS} \
    "${ROOT_DIR}/build/obj/kernels.o" \
    "${ROOT_DIR}/build/obj/main.o" \
    ${CUDA_LIB_DIRS} ${CUDA_LIB} \
    -o "${ROOT_DIR}/build/bin/main"

echo "Binary: ${ROOT_DIR}/build/bin/main"
