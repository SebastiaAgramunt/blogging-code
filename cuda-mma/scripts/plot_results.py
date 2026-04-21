#!/usr/bin/env python3
from path import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = Path(__file__).parent.parent / "output"

kernels = {
    "Naive":     "naive.csv",
    "Tiled":     "tiled.csv",
    "Coalesced": "coalesced.csv",
    "cuBLAS":    "cublas.csv",
    "CBLAS":     "cblas.csv",
}

def load(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("CUDA SGEMM Roofline Analysis - Nvidia GPU A6000")
ax_gflops, ax_bw = axes

for label, filename in kernels.items():
    df = load(filename)
    if df is None:
        print(f"Warning: {filename} not found, skipping.")
        continue
    ax_gflops.plot(df["size"], df["gflops"],        marker="o", label=label)
    ax_bw.plot    (df["size"], df["bandwidth_gbs"], marker="o", label=label)

def ai_fmt(x, _):
    if x >= 1e9: return f"{x/1e9:.3g}G"
    if x >= 1e6: return f"{x/1e6:.3g}M"
    if x >= 1e3: return f"{x/1e3:.3g}k"
    return f"{x:.3g}"

# AI = S/8 FLOP/Byte = S/8 * 1e6 FLOP/MB  (2S³ flops / 16S² bytes)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Matrix size (S×S)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    sec = ax.secondary_xaxis("top", functions=(lambda x: x / 8 * 1e6, lambda x: x * 8 / 1e6))
    sec.set_xscale("log", base=2)
    sec.xaxis.set_major_formatter(ticker.FuncFormatter(ai_fmt))
    sec.set_xlabel("Arithmetic Intensity (FLOP/MB)")

ax_gflops.set_title("Throughput")
ax_gflops.set_ylabel("GFLOP/s")

ax_bw.set_title("Memory Bandwidth")
ax_bw.set_ylabel("GB/s")

fig.tight_layout()
out_path = OUTPUT_DIR / "roofline.png"
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
plt.show()
