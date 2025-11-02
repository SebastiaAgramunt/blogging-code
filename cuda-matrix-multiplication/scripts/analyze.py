#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

import logging
import matplotlib as mpl
import matplotlib.image as mpimg
from PIL import Image

THIS_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = THIS_DIR.parent.resolve()


def plot_cpu_vs_gpu(df: pd.DataFrame, max_n_cpu=1024, xrange=[0, 100], yrange=[0, 5], filename="gpu_cpu_performance.png"):
    fig, ax = plt.subplots(1, 1, dpi=100, layout='constrained', figsize=(15, 8))
    
    # subset CPU, we don't calculate every N, takes way too much time
    df_tmp = df.loc[df["N"] <= max_n_cpu, :]
    plt.plot(df_tmp["N"], df_tmp["compute_time_cpu"], marker="o", label="CPU time", lw=5, color="black")
    plt.plot(df["N"], df["total_time_gpu"], marker="o", label="GPU time", lw=5, color="green")
    
    plt.xlabel("Matrix Size $N$", fontsize=16)
    plt.ylabel("Time ($ms$)", fontsize=16)
    plt.title("Computation time as function of square matrix size", fontsize=18)
    
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    # ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=20)
    ax.grid(True, which="both", ls="--", lw=2.0, alpha=0.5)
    fig.savefig(ROOT_DIR / "results" / filename)
    plt.close(fig)
    
def plot_gpu_times(df: pd.DataFrame, xrange=[0, 100], yrange=[0, 5], filename="gpu_comparison_times.png"):
    fig, ax = plt.subplots(1, 1, dpi=100, layout='constrained', figsize=(15, 8))
    plt.plot(df["N"], df["compute_time_gpu"], marker="o", label="GPU time simple", lw=5, color="green")
    plt.plot(df["N"], df["total_time_gpu_tiled"], marker="o", label="GPU time tiled", lw=5, color="blue")
    plt.plot(df["N"], df["total_time_gpu_cublas"], marker="o", label="GPU time cuBLAS", lw=5, color="gray")
    
    plt.xlabel("Matrix Size $N$", fontsize=16)
    plt.ylabel("Time ($ms$)", fontsize=16)
    plt.title("Computation time on GPU per method", fontsize=18)
    
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.legend(fontsize=20)
    ax.grid(True, which="both", ls="--", lw=2.0, alpha=0.5)
    fig.savefig(ROOT_DIR / "results" / filename)
    plt.close(fig)
    
def main():
    df = pd.read_csv(ROOT_DIR / "results" / "output.csv")
    plot_cpu_vs_gpu(df)
    plot_cpu_vs_gpu(df, xrange=[0, 10240], yrange=[0, 800], filename="gpu_cpu_performance_large.png")
    plot_gpu_times(df)
    plot_gpu_times(df, xrange=[0, 10240], yrange=[0, 800], filename="gpu_comparison_times_large.png")


if __name__ == "__main__":
    main()