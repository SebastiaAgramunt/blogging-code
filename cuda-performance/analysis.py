#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv("performance_time_threads_256_m_1_dtype_float.csv")
# df = pd.read_csv("performance_time_threads_256_m_4_dtype_float.csv")
# df = pd.read_csv("performance_time_threads_256_m_20_dtype_float.csv")
df = pd.read_csv("performance_time_threads_256_m_40_dtype_float.csv")

# Use two distinct colors from the viridis colormap
cmap = plt.cm.viridis
gpu_color = cmap(0.2)   # bright greenish
cpu_color = cmap(0.8)   # purple/blue

fig, ax = plt.subplots(1, 2, dpi=180, layout='constrained', figsize=(15, 5))

# Left plot: log2 scale
ax[0].plot(np.log2(df["N"]), df["totalTime(ms)"], label="GPU Time", marker='o', color=gpu_color, linewidth=2)
ax[0].plot(np.log2(df["N"]), df["cpuTime(ms)"], label="CPU Time", marker='o', color=cpu_color, linewidth=2)
ax[0].set_xlabel("Size (N, log2 scale)")
ax[0].set_ylabel("Time (ms)")
ax[0].set_xlim(0, 30)
ax[0].set_ylim(-200, 1000)
ax[0].grid(True, which="both", ls="--", lw=0.5)
ax[0].legend()

# # Right plot: linear scale
# ax[0][1].plot(df["Mb"], df["totalTime(ms)"], label="GPU Time", marker='o', color=gpu_color, linewidth=2)
# ax[0][1].plot(df["Mb"], df["cpuTime(ms)"], label="CPU Time", marker='o', color=cpu_color, linewidth=2)
# ax[0][1].set_xlabel("Size (Mb)")
# ax[0][1].set_ylabel("Time (ms)")
# # ax[0][1].set_xticks(np.arange(0, df["Mb"].max()+1, step=50))
# ax[0][1].set_xlim(0, 4096)
# ax[0][1].set_ylim(-1000, 10000)
# ax[0][1].grid(True, which="both", ls="--", lw=0.5)
# ax[0][1].legend()


ax[1].plot(np.log2(df["N"]), df["totalTime(ms)"], label="GPU Time", marker='o', color=gpu_color, linewidth=2)
ax[1].plot(np.log2(df["N"]), df["cpuTime(ms)"], label="CPU Time", marker='o', color=cpu_color, linewidth=2)
ax[1].set_xlabel("Size (N, log2 scale)")
ax[1].set_ylabel("Time (ms)")
ax[1].set_xlim(20, 30)
ax[1].set_ylim(-0.2, 600.0)
ax[1].grid(True, which="both", ls="--", lw=0.5)
ax[1].legend()


# # Right plot: linear scale
# ax[1][1].plot(df["Mb"], df["totalTime(ms)"], label="GPU Time", marker='o', color=gpu_color, linewidth=2)
# ax[1][1].plot(df["Mb"], df["cpuTime(ms)"], label="CPU Time", marker='o', color=cpu_color, linewidth=2)
# ax[1][1].set_xlabel("Size (Mb)")
# ax[1][1].set_ylabel("Time (ms)")
# # ax[0][1].set_xticks(np.arange(0, df["Mb"].max()+1, step=50))
# ax[1][1].set_xlim(0, 0.5)
# ax[1][1].set_ylim(-0.1, 1)
# ax[1][1].grid(True, which="both", ls="--", lw=0.5)
# ax[1][1].legend()

fig.suptitle("Performance Analysis", fontsize=14, fontweight="bold")
fig.savefig("performance_analysis.png")
plt.close()


fig, ax = plt.subplots(1, 1, dpi=180, layout='constrained', figsize=(15, 5))
ax.plot(np.log2(df["N"]), df["allocateTime(ms)"], label="Allocate Time", marker='o', color='blue', linewidth=2)
ax.plot(np.log2(df["N"]), df["loadTime(ms)"], label="Load Time", marker='o', color='orange', linewidth=2)
ax.plot(np.log2(df["N"]), df["calcTime(ms)"], label="Calc Time", marker='o', color='green', linewidth=2)
ax.plot(np.log2(df["N"]), df["loadTimeBack(ms)"], label="Load Back Time", marker='o', color='red', linewidth=2)
ax.set_xlabel("Size (N, log2 scale)")
ax.set_ylabel("Time (ms)")
ax.set_xlim(25, 30)
ax.set_ylim(-100, 500)
ax.grid(True, which="both", ls="--", lw=0.5)
ax.legend()
fig.suptitle("Breakdown of GPU Time", fontsize=14, fontweight="bold")
fig.savefig("gpu_time_breakdown.png")
plt.close()

