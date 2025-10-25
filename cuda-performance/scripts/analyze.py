#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

THIS_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = THIS_DIR.parent.resolve()


def plot_cpu_vs_gpu(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, dpi=100, layout='constrained', figsize=(15, 8))
    
    additions = [1, 256, 4096, 16384]
    tags = ["1", "$2^8$", "$2^{11}$", "$2^{14}$"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(additions)))
    for i, key in enumerate(additions):
        df_tmp = df.loc[df["number_of_additions"] == key, :]
        plt.plot(np.log10(df_tmp["N"]), df_tmp["total_GPU_time"], marker='o', label=f"GPU aditions={tags[i]}", color=colors[i], lw=3)

    number_of_additions = 1
    df_tmp = df.loc[df["number_of_additions"]==number_of_additions, :]
    plt.plot(np.log10(df_tmp["N"]), df_tmp["total_CPU_time"], marker='o', label=f"CPU aditions={number_of_additions}", color="black", lw=5)
    
    number_of_additions = 256
    df_tmp = df.loc[df["number_of_additions"]==number_of_additions, :]
    plt.plot(np.log10(df_tmp["N"]), df_tmp["total_CPU_time"], marker='o', label=f"CPU aditions=$2^8$", color="gray", lw=5)
    
    plt.xlabel("$\log_{10}(N)$", fontsize=16)
    plt.ylabel("Time ($ms$)", fontsize=16)
    plt.title("Computation time as function of number of elements in the vectors", fontsize=18)
    plt.grid(True)
    
    ax.set_xlim(4, 9)
    ax.set_ylim(-110, 1500)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=20)
    ax.grid(True, which="both", ls="--", lw=2.0, alpha=0.5)
    fig.savefig(ROOT_DIR / "results" / "gpu_cpu_performance.png")
    plt.close(fig)
    
    
def plot_cpu_vs_gpu_1(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, dpi=100, layout='constrained', figsize=(15, 8))
    
    number_of_additions = 1
    df_tmp = df.loc[df["number_of_additions"]==number_of_additions, :]
    
    plt.plot(np.log10(df_tmp["N"]), df_tmp["total_GPU_time"], marker='o', label=f"GPU", color="green", lw=3)
    plt.plot(np.log10(df_tmp["N"]), df_tmp["total_CPU_time"], marker='o', label=f"CPU", color="black", lw=3)
    plt.vlines(x=6.5, ymin=-1.0, ymax=25, color="gray", lw=4)
    ax.text(x=6.15, y=6, s="vector length\n~3.1M floats", fontsize=14, alpha=1)
    ax.text(x=6.15, y=22.5, s="CPU is faster\nthan GPU", fontsize=14, alpha=1)
    ax.text(x=6.65, y=22.5, s="GPU is faster\nthan CPU", fontsize=14, alpha=1)
    
    plt.xlabel("$\log_{10}(N)$", fontsize=16)
    plt.ylabel("Time ($ms$)", fontsize=16)
    plt.title("Total time for one vector adition", fontsize=18)
    plt.grid(True)
    # plt.tight_layout()
   
    ax.set_xlim(5, 8)
    ax.set_ylim(-1, 25)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=20)
    ax.grid(True, which="both", ls="--", lw=2.0, alpha=0.5)
    fig.savefig(ROOT_DIR / "results" / "gpu_cpu_performance_1.png")
    plt.close(fig)

def plot_gpu_times(df, mult_times=2048):
    
    # df_tmp = df.loc[df["number_of_additions"] == 1, :]
    df_tmp = df.loc[df["number_of_additions"] == mult_times, :]
    # df_tmp = df.loc[df["number_of_additions"] == 2048, :]
    x = np.log10(df_tmp["N"].to_numpy())
    alloc = df_tmp["allocate_time"].to_numpy()
    copy_h2d = df_tmp["copy_H2D_time"].to_numpy()
    compute = df_tmp["compute_time"].to_numpy()
    copy_d2h = df_tmp["copy_D2H_time"].to_numpy()
    free_ = df_tmp["free_time"].to_numpy()

    series = [alloc, copy_h2d, compute, copy_d2h, free_]
    labels = ["Allocation", "Copy H→D", "Compute", "Copy D→H", "Free"]

    fig, ax = plt.subplots(1, 1, dpi=100, layout='constrained', figsize=(15, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(series)))

    ax.stackplot(
        x,
        series,
        labels=labels,
        colors=colors,
        linewidth=0.0,
        alpha=0.9,
    )

    # # overlay total time as a line (use your column if you have it)
    total = df_tmp["total_time_gpu"].to_numpy() if "total_time_gpu" in df_tmp.columns else np.sum(series, axis=0)
    ax.plot(x, total, label="Total GPU Time", color="red", lw=4)
    
    # # plot cpu time but filter by small times 
    # cpu_time = df_tmp.loc[df["N"] < 1024, "compute_time"].to_numpy()
    # cpu_time_x = df_tmp.loc[df["N"] < 1024, "N"].to_numpy()
    # ax.plot(cpu_time_x, cpu_time, label="CPU Time", lw=4, color="black")

    # ax.set_xlim(xrange[0], xrange[1])
    # ax.set_ylim(yrange[0], yrange[1])
    # ax.set_xlim(0, 1.0e9)
    ax.set_ylim(0, 1000)
    ax.set_xlim(6, 9)
    ax.set_title(r"vector addition " + f"{mult_times} times", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel("$\log_{10} N$", fontsize=16)
    ax.set_ylabel("time (ms)", fontsize=16)
    ax.legend(fontsize=16, loc="best", ncol=2, frameon=True)
    ax.grid(True, which="both", ls="--", lw=2.0, alpha=0.5)

    filename=f"gpu_times_stacked_{mult_times}.png"
    fig.savefig(ROOT_DIR / "results" / filename)
    plt.close(fig)
    
def plot_percentages_bar(df, number_of_additions=1, N=512):
    metrics = ["allocate_time", "copy_H2D_time", "compute_time", "copy_D2H_time", "free_time"]
    times = df.loc[(df["N"] == N) & (df["number_of_additions"]==number_of_additions), metrics].values.flatten()
    
    total = sum(times)

    fig, ax = plt.subplots(figsize=(5,8))

    total = sum(times)
    bottom = 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))

    for metric, time, color in zip(metrics, times, colors):
        ax.bar(0, time, bottom=bottom, color=color, width=0.6,
            label=f"{metric}: {(time/total)*100:.1f}%")
        bottom += time

    leg = ax.legend(
        title="Times",
        loc="upper left",
        bbox_to_anchor=(1.1, 1.0),   # just outside the axes
        fontsize=9,
        title_fontsize=10,
        frameon=False,
    )

    # Make room on the right for the legend
    fig.subplots_adjust(right=0.5)   # tweak 0.72–0.85 as needed

    # Optional clean-up
    ax.set_xticks([0])
    ax.set_xticklabels(["Total Time"])
    ax.set_ylabel("Time (ms)")
    ax.set_title(r"GPU Time Percentages $log_{10}$" +f"\nN={int(np.floor(np.log10(N)))}, additions {number_of_additions}")

    plt.tight_layout()
    filename=f"percentages_performance_N_{N}additions_{number_of_additions}.png"
    fig.savefig(ROOT_DIR / "results" / filename)
    plt.close(fig)

def main():
    df = pd.read_csv(ROOT_DIR / "results" / "results.csv")
    plot_cpu_vs_gpu_1(df)
    plot_cpu_vs_gpu(df)
    plot_gpu_times(df, mult_times=1)
    plot_gpu_times(df, mult_times=1024)
    plot_gpu_times(df, mult_times=2048)
    plot_percentages_bar(df, number_of_additions=2048, N=1073741824)
    plot_percentages_bar(df, number_of_additions=2048, N=512)

if __name__ == "__main__":
    main()