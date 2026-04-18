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
}

def load(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("CUDA SGEMM Roofline Analysis - Nvidia GPU A10")
ax_gflops, ax_bw, ax_ai = axes

for label, filename in kernels.items():
    df = load(filename)
    if df is None:
        print(f"Warning: {filename} not found, skipping.")
        continue
    ax_gflops.plot(df["size"], df["gflops"],          marker="o", label=label)
    ax_bw.plot    (df["size"], df["bandwidth_gbs"],   marker="o", label=label)
    ax_ai.plot    (df["size"], df["arithmetic_intensity"], marker="o", label=label)

for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Matrix size (S×S)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

ax_gflops.set_title("Throughput")
ax_gflops.set_ylabel("GFLOP/s")

ax_bw.set_title("Memory Bandwidth")
ax_bw.set_ylabel("GB/s")

ax_ai.set_title("Arithmetic Intensity")
ax_ai.set_ylabel("FLOP/Byte")

fig.tight_layout()
out_path = OUTPUT_DIR / "roofline.png"
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
plt.show()
