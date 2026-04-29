#!/usr/bin/env python3
from path import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = Path(__file__).parent.parent / "output"

kernels = {
    # "cBLAS":     "cblas.csv",
    "Naive":     "naive.csv",
    "Coalesced": "coalesced.csv",
    "Tiled":     "tiled.csv",
    "Coarsened": "coarsened.csv",
    "cuBLAS":    "cublas.csv",
    "CUTLASS":   "cutlass.csv",
}

colors = {
    "Naive":     "C0",
    "Coalesced": "C1",
    "Tiled":     "C2",
    "Coarsened": "C3",
    "cuBLAS":    "C4",
    "cBLAS":     "C6",
    "CUTLASS":   "C5",
}

def load(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("CUDA SGEMM Roofline Analysis - Nvidia GPU A100")
ax_gflops, ax_bw = axes

for label, filename in kernels.items():
    df = load(filename)
    if df is None:
        print(f"Warning: {filename} not found, skipping.")
        continue
    ax_gflops.plot(df["size"], df["gflops"],        marker="o", label=label, color=colors[label])
    ax_bw.plot    (df["size"], df["bandwidth_gbs"], marker="o", label=label, color=colors[label])

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

# --- Naive vs cBLAS throughput comparison ---
naive_vs_cblas = {
    "Naive": "naive.csv",
    "cBLAS": "cblas.csv",
}

fig2, ax2 = plt.subplots(figsize=(7, 5))
fig2.suptitle("SGEMM Roofline Analysis - Nvidia GPU A100 vs AMD EPYC 7J13 64-Core")

for label, filename in naive_vs_cblas.items():
    df = load(filename)
    if df is None:
        print(f"Warning: {filename} not found, skipping.")
        continue
    ax2.plot(df["size"], df["gflops"], marker="o", label=label, color=colors[label])

ax2.set_xscale("log", base=2)
ax2.set_yscale("log")
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax2.set_xlabel("Matrix size (S×S)")
ax2.set_ylabel("GFLOP/s")
ax2.set_title("Throughput")
ax2.legend()
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

sec2 = ax2.secondary_xaxis("top", functions=(lambda x: x / 8 * 1e6, lambda x: x * 8 / 1e6))
sec2.set_xscale("log", base=2)
sec2.xaxis.set_major_formatter(ticker.FuncFormatter(ai_fmt))
sec2.set_xlabel("Arithmetic Intensity (FLOP/MB)")

fig2.tight_layout()
out_path2 = OUTPUT_DIR / "roofline_naive_cblas.png"
plt.savefig(out_path2, dpi=150)
print(f"Saved {out_path2}")
plt.show()
