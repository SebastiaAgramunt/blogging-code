#!/usr/bin/env bash
# Downloads vendored dependencies into vendor/.
# Re-running is safe: existing checkouts are skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor"

CUTLASS_TAG="v4.4.2"
CUTLASS_DIR="$VENDOR_DIR/cutlass"

echo "==> Downloading CUTLASS $CUTLASS_TAG into $CUTLASS_DIR ..."

if [ -d "$CUTLASS_DIR/.git" ]; then
    echo "    Already present, skipping."
else
    mkdir -p "$VENDOR_DIR"
    git clone \
        --filter=blob:none \
        --sparse \
        --depth=1 \
        --branch "$CUTLASS_TAG" \
        https://github.com/NVIDIA/cutlass.git \
        "$CUTLASS_DIR"
    git -C "$CUTLASS_DIR" sparse-checkout set include
fi

echo "Done. vendor/ is ready."
