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

fig, axes = plt.subplots(2, 1, figsize=(8, 12))
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

def plot_comparison(series, title, out_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(title)

    for label, filename in series.items():
        df = load(filename)
        if df is None:
            print(f"Warning: {filename} not found, skipping.")
            continue
        ax.plot(df["size"], df["gflops"], marker="o", label=label, color=colors[label])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Matrix size (S×S)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Throughput")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    sec = ax.secondary_xaxis("top", functions=(lambda x: x / 8 * 1e6, lambda x: x * 8 / 1e6))
    sec.set_xscale("log", base=2)
    sec.xaxis.set_major_formatter(ticker.FuncFormatter(ai_fmt))
    sec.set_xlabel("Arithmetic Intensity (FLOP/MB)")

    fig.tight_layout()
    out_path = OUTPUT_DIR / out_name
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()

# --- Naive vs cBLAS throughput comparison ---
plot_comparison(
    {"Naive": "naive.csv", "cBLAS": "cblas.csv"},
    "SGEMM Roofline Analysis - Nvidia GPU A100 vs AMD EPYC 7J13 64-Core",
    "roofline_naive_cblas.png",
)

# --- Coalesced vs Tiled throughput comparison ---
plot_comparison(
    {"Coalesced": "coalesced.csv", "Tiled": "tiled.csv"},
    "SGEMM Roofline Analysis - Nvidia GPU A100",
    "roofline_coalesced_tiled.png",
)

# --- Naive vs Coalesced throughput comparison ---
plot_comparison(
    {"Naive": "naive.csv", "Coalesced": "coalesced.csv"},
    "SGEMM Roofline Analysis - Nvidia GPU A100",
    "roofline_naive_coalesced.png",
)

# --- Tiled vs cuBLAS throughput comparison ---
plot_comparison(
    {"Tiled": "tiled.csv", "cuBLAS": "cublas.csv"},
    "SGEMM Roofline Analysis - Nvidia GPU A100",
    "roofline_tiled_cublas.png",
)

# --- cuBLAS vs CUTLASS throughput comparison ---
plot_comparison(
    {"cuBLAS": "cublas.csv", "CUTLASS": "cutlass.csv"},
    "SGEMM Roofline Analysis - Nvidia GPU A100",
    "roofline_cublas_cutlass.png",
)
