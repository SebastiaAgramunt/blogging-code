#!/usr/bin/env python3
"""
analyze.py — Visualise memory-bound vs compute-bound GPU behaviour.

Reads:
  results/roofline.csv
  results/bw_saturation.csv
  results/device_info.csv

Produces:
  results/roofline.png          — classic roofline model
  results/bw_vs_m.png           — bandwidth & GFLOP/s vs m
  results/bw_saturation.png     — bandwidth vs vector size (N)
  results/time_breakdown.png    — compute time vs m with regime labels
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

THIS_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = THIS_DIR.parent.resolve()
RESULTS  = ROOT_DIR / "results"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 120,
})

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_device_info(path: pathlib.Path) -> dict:
    df = pd.read_csv(path, index_col="key")
    return {k: float(v) if v.replace(".", "", 1).isdigit() else v
            for k, v in df["value"].items()}


def load_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run execute.sh first")
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Roofline model
# ─────────────────────────────────────────────────────────────────────────────

def plot_roofline(df: pd.DataFrame, info: dict):
    """
    Classic roofline: GFLOP/s on y-axis, arithmetic intensity on x-axis.

    Two hardware ceilings:
      • Memory-bandwidth ceiling: GFLOP/s = BW_GB/s × AI
      • Compute ceiling:           GFLOP/s = peak_FP32

    Points below the ridge are memory-bound; above are compute-bound.
    """
    peak_bw    = info["theoretical_bw_gbs"]      # GB/s
    peak_flops = info["theoretical_fp32_gflops"]  # GFLOP/s
    emp_bw     = info.get("empirical_bw_gbs", peak_bw)
    ridge_pt   = info["ridge_point_flop_per_byte"]

    ai_range = np.logspace(-2, 4, 500)

    # Roofline ceiling
    roof_mem     = peak_bw    * ai_range   # memory-bandwidth ceiling (GFLOP/s)
    roof_compute = np.full_like(ai_range, peak_flops)
    roofline     = np.minimum(roof_mem, roof_compute)

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    ax.loglog(ai_range, roofline, "k-",  lw=2.5, label="Roofline (theoretical)", zorder=3)
    ax.loglog(ai_range, emp_bw * ai_range, "k--", lw=1.5,
              label=f"Empirical BW ({emp_bw:.0f} GB/s)", zorder=3)
    ax.axhline(peak_flops, color="steelblue", lw=1.5, ls=":",
               label=f"Peak FP32 ({peak_flops:.0f} GFLOP/s)")

    # Mark ridge point
    ax.axvline(ridge_pt, color="gray", lw=1.2, ls="--")
    ax.text(ridge_pt * 1.08, peak_flops * 0.55, f"Ridge\n{ridge_pt:.1f} FLOP/byte",
            fontsize=10, color="gray", va="center")

    # Shade regimes
    ax.axvspan(ai_range[0], ridge_pt, alpha=0.06, color="royalblue",
               label="Memory-bound region")
    ax.axvspan(ridge_pt, ai_range[-1], alpha=0.06, color="tomato",
               label="Compute-bound region")

    # Measured points, coloured by m
    m_vals = df["m"].values
    sc = ax.scatter(df["arithmetic_intensity"], df["achieved_gflops"],
                    c=np.log2(m_vals.astype(float)), cmap="plasma",
                    s=70, zorder=5, edgecolors="k", linewidths=0.4)

    # Annotate a few interesting m values
    for m_label in [1, 32, 128, 1024, 8192]:
        row = df[df["m"] == m_label]
        if row.empty:
            continue
        ax.annotate(f"m={m_label}",
                    xy=(row["arithmetic_intensity"].values[0],
                        row["achieved_gflops"].values[0]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color="black")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("log₂(m)  — FMA iterations per element")

    device = info.get("device_name", "GPU")
    ax.set_xlabel("Arithmetic Intensity  [FLOP / byte]")
    ax.set_ylabel("Achieved Throughput  [GFLOP/s]")
    ax.set_title(f"Roofline Model — {device}\nvector_add + m FMAs per element")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlim(0.05, 5000)
    ax.set_ylim(1, peak_flops * 3)

    fig.savefig(RESULTS / "roofline.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: roofline.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Bandwidth and GFLOP/s vs m
# ─────────────────────────────────────────────────────────────────────────────

def plot_bw_vs_m(df: pd.DataFrame, info: dict):
    """
    Two y-axes: left = achieved GB/s, right = achieved GFLOP/s.
    Dashed horizontal lines show hardware ceilings.
    The cross-over from BW-saturation to FLOP-saturation is the ridge point.
    """
    peak_bw    = info["theoretical_bw_gbs"]
    peak_flops = info["theoretical_fp32_gflops"]
    emp_bw     = info.get("empirical_bw_gbs", peak_bw)

    fig, ax1 = plt.subplots(figsize=(11, 6), layout="constrained")
    ax2 = ax1.twinx()

    m = df["m"].values

    ln1, = ax1.semilogx(m, df["achieved_bw_gbs"],  "o-", color="royalblue",
                        lw=2, ms=5, label="Achieved bandwidth")
    ln2, = ax2.semilogx(m, df["achieved_gflops"],  "s-", color="tomato",
                        lw=2, ms=5, label="Achieved GFLOP/s")

    ax1.axhline(emp_bw,     color="royalblue", ls="--", lw=1.5,
                label=f"Empirical peak BW ({emp_bw:.0f} GB/s)")
    ax2.axhline(peak_flops, color="tomato",    ls="--", lw=1.5,
                label=f"Peak FP32 ({peak_flops:.0f} GFLOP/s)")

    # Regime annotations
    ax1.axvspan(m[0], m[-1] // 4, alpha=0.05, color="royalblue")
    ax1.axvspan(m[-1] // 4, m[-1], alpha=0.05, color="tomato")
    ax1.text(1.5, emp_bw * 0.6, "← Memory-\nbound", fontsize=10,
             color="royalblue", alpha=0.8)
    ax1.text(m[-1] * 0.3, emp_bw * 0.6, "Compute-\nbound →", fontsize=10,
             color="tomato", alpha=0.8)

    ax1.set_xlabel("m  (FMA iterations per element, log scale)")
    ax1.set_ylabel("Achieved Bandwidth  [GB/s]", color="royalblue")
    ax2.set_ylabel("Achieved Throughput  [GFLOP/s]", color="tomato")
    ax1.tick_params(axis="y", colors="royalblue")
    ax2.tick_params(axis="y", colors="tomato")

    lines  = [ln1, ln2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="center left")

    device = info.get("device_name", "GPU")
    ax1.set_title(f"Bandwidth & Compute Throughput vs Arithmetic Intensity — {device}")
    ax1.grid(True, which="both", ls="--", lw=0.6, alpha=0.4)

    fig.savefig(RESULTS / "bw_vs_m.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: bw_vs_m.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Bandwidth saturation vs vector size
# ─────────────────────────────────────────────────────────────────────────────

def plot_bw_saturation(df: pd.DataFrame, info: dict):
    """
    For m=1 (pure memory-bound kernel), shows that small N can't saturate
    HBM bandwidth — you need enough concurrent memory requests.
    """
    peak_bw = info["theoretical_bw_gbs"]
    emp_bw  = info.get("empirical_bw_gbs", peak_bw)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

    ax.semilogx(df["N"], df["achieved_bw_gbs"], "o-", color="royalblue",
                lw=2, ms=5, label="Achieved bandwidth")
    ax.axhline(emp_bw,   color="royalblue", ls="--", lw=1.5,
               label=f"Empirical peak BW ({emp_bw:.0f} GB/s)")
    ax.axhline(peak_bw,  color="gray",      ls=":",  lw=1.2,
               label=f"Theoretical peak BW ({peak_bw:.0f} GB/s)")

    ax.set_xlabel("Vector length  N  (log scale)")
    ax.set_ylabel("Achieved Bandwidth  [GB/s]")
    device = info.get("device_name", "GPU")
    ax.set_title(f"Memory Bandwidth Saturation (m=1) — {device}\n"
                 f"Bandwidth rises as N grows, saturating once enough warps fill the SM pipeline")
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    fig.savefig(RESULTS / "bw_saturation.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: bw_saturation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Kernel time vs m with regime labels
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_vs_m(df: pd.DataFrame, info: dict):
    """
    Log-log plot of kernel time vs m.
      • Memory-bound regime:  time is roughly constant (bottleneck is BW)
      • Compute-bound regime: time ∝ m  (slope ≈ 1 on log-log)
    """
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

    m = df["m"].values
    t = df["kernel_time_ms"].values

    ax.loglog(m, t, "o-", color="darkorchid", lw=2, ms=5)

    # Ideal memory-bound reference (flat line at t[0])
    ax.axhline(t[0], color="royalblue", ls="--", lw=1.5, alpha=0.7,
               label=f"Memory-bound limit  ({t[0]:.2f} ms)")

    # Ideal compute-bound reference: t ∝ m, anchored at last measured point
    m_ref = m[-1]
    t_ref = t[-1]
    m_line = np.array([m[len(m)//3], m[-1]], dtype=float)
    t_line = t_ref * (m_line / m_ref)
    ax.loglog(m_line, t_line, color="tomato", ls="--", lw=1.5, alpha=0.7,
              label="Compute-bound (slope 1)")

    ax.set_xlabel("m  (FMA iterations per element)")
    ax.set_ylabel("Kernel time  [ms]")
    device = info.get("device_name", "GPU")
    ax.set_title(f"Kernel Time vs m — {device}\n"
                 "Flat = memory-bound; rising linearly (slope 1 on log-log) = compute-bound")
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.5)

    fig.savefig(RESULTS / "time_vs_m.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: time_vs_m.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Efficiency heatmap (bandwidth util + compute util vs m)
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficiency(df: pd.DataFrame, info: dict):
    """
    Normalised utilisation:
      bw_util   = achieved_bw   / empirical_peak_bw  ∈ [0, 1]
      flop_util = achieved_gflops / peak_fp32         ∈ [0, 1]
    """
    emp_bw    = info.get("empirical_bw_gbs", info["theoretical_bw_gbs"])
    peak_flops = info["theoretical_fp32_gflops"]

    bw_util   = df["achieved_bw_gbs"] / emp_bw
    flop_util = df["achieved_gflops"] / peak_flops
    m         = df["m"].values

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

    ax.semilogx(m, bw_util   * 100, "o-", color="royalblue", lw=2, ms=5,
                label="Memory BW utilisation (%)")
    ax.semilogx(m, flop_util * 100, "s-", color="tomato",    lw=2, ms=5,
                label="FP32 compute utilisation (%)")

    ax.axhline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_ylim(0, 115)
    ax.set_xlabel("m  (FMA iterations per element)")
    ax.set_ylabel("Hardware Utilisation  [%]")
    device = info.get("device_name", "GPU")
    ax.set_title(f"Hardware Utilisation vs Arithmetic Intensity — {device}")
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.4)

    fig.savefig(RESULTS / "efficiency.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: efficiency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    info      = load_device_info(RESULTS / "device_info.csv")
    df_roof   = load_csv("roofline.csv")
    df_bw_sat = load_csv("bw_saturation.csv")

    print(f"Device: {info.get('device_name', '?')}")
    print(f"Theoretical BW:    {info['theoretical_bw_gbs']:.0f} GB/s")
    print(f"Empirical BW:      {info.get('empirical_bw_gbs', float('nan')):.0f} GB/s")
    print(f"Peak FP32:         {info['theoretical_fp32_gflops']:.0f} GFLOP/s")
    print(f"Ridge point:       {info['ridge_point_flop_per_byte']:.1f} FLOP/byte\n")

    plot_roofline(df_roof, info)
    plot_bw_vs_m(df_roof, info)
    plot_time_vs_m(df_roof, info)
    plot_efficiency(df_roof, info)
    plot_bw_saturation(df_bw_sat, info)


if __name__ == "__main__":
    main()
