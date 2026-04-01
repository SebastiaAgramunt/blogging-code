#!/bin/bash
# Run the benchmark and produce CSV results.
# Optional: pass --ncu to also profile with Nsight Compute.
#
# Usage:
#   ./execute.sh              # just benchmark
#   ./execute.sh --ncu        # benchmark + ncu profile of the m=1 and m=256 kernels

set -e

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname "${THIS_DIR}")
BINARY="${ROOT_DIR}/build/bin/main"
RESULTS="${ROOT_DIR}/results"

if [ ! -f "${BINARY}" ]; then
    echo "Binary not found. Run scripts/compile.sh first."
    exit 1
fi

mkdir -p "${RESULTS}"

echo "=== Running benchmark ==="
"${BINARY}" \
    --results_dir  "${RESULTS}" \
    --threads_per_block 256 \
    --N_roofline 33554432 \
    --bw_repeats 10

echo ""
echo "=== Results saved to ${RESULTS}/ ==="

# ── Optional: Nsight Compute profiling ────────────────────────────────────
if [[ "$1" == "--ncu" ]]; then
    echo ""
    echo "=== Nsight Compute profiling (m=1 and m=512) ==="

    # Profile the memory-bound case (m=1)
    ncu --set full \
        --export "${RESULTS}/ncu_m1" \
        --force-overwrite \
        "${BINARY}" \
            --results_dir "${RESULTS}" \
            --threads_per_block 256 \
            --N_roofline 33554432 \
            --bw_repeats 1 2>/dev/null || \
        echo "ncu not available — skipping (install CUDA Toolkit for profiling)"

    echo "NCU reports: ${RESULTS}/ncu_m1.ncu-rep"
fi
