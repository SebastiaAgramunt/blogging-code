#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_1 = pd.read_csv("../outputs/performance_time_threads_256_m_1_dtype_float.csv")
df_4 = pd.read_csv("../outputs/performance_time_threads_256_m_4_dtype_float.csv")
df_20= pd.read_csv("../outputs/performance_time_threads_256_m_20_dtype_float.csv")
df_40 = pd.read_csv("../outputs/performance_time_threads_256_m_40_dtype_float.csv")

# Use two distinct colors from the viridis colormap
cmap = plt.cm.viridis
cpu_color_1 = "black"
cpu_color_4 = "gray"
gpu_colors = [cmap(x) for x in [0.15, 0.35, 0.55, 0.75]]
fig, ax = plt.subplots(1, 1, dpi=100, layout='constrained', figsize=(15, 8))

ax.plot(np.log2(df_1["N"]),  df_1["calcTime(ms)"],  label="GPU Time 1-OP",  marker='o', color=gpu_colors[0], linewidth=4, alpha=0.8)
ax.plot(np.log2(df_4["N"]),  df_4["calcTime(ms)"],  label="GPU Time 4-OP",  marker='o', color=gpu_colors[1], linewidth=4, alpha=0.8)
ax.plot(np.log2(df_20["N"]), df_20["calcTime(ms)"], label="GPU Time 20-OP", marker='o', color=gpu_colors[2], linewidth=4, alpha=0.8)
ax.plot(np.log2(df_40["N"]), df_40["calcTime(ms)"], label="GPU Time 40-OP", marker='o', color=gpu_colors[3], linewidth=4, alpha=0.8)
ax.plot(np.log2(df_1["N"]), df_1["cpuTime(ms)"], label="CPU Time 1-OP", marker='o', color=cpu_color_1, linewidth=4, alpha=0.8)
ax.plot(np.log2(df_4["N"]), df_4["cpuTime(ms)"], label="CPU Time 4-OP", marker='o', color=cpu_color_4, linewidth=4, alpha=0.8)

ax.tick_params(axis="both", which="major", labelsize=14)
ax.set_xlabel("Size (N, log2 scale)", fontsize=14)
ax.set_ylabel("Time (ms)", fontsize=14)
ax.set_xlim(15, 30)
ax.set_ylim(0, 10)
ax.grid(True, which="both", ls="--", lw=2.0, alpha=0.5)
ax.legend(fontsize=14)

fig.suptitle("Calculation Time", fontsize=18)
fig.savefig("calculation_time.png")
plt.close()


fig, ax = plt.subplots(1, 1, dpi=100, layout='constrained', figsize=(15, 8))
ax.plot(np.log2(df_1["N"]),  df_1["totalTime(ms)"],  label="GPU Time 1-OP",  marker='o', color=gpu_colors[0], linewidth=4, alpha=0.8)
ax.plot(np.log2(df_4["N"]),  df_4["totalTime(ms)"],  label="GPU Time 4-OP",  marker='o', color=gpu_colors[1], linewidth=4, alpha=0.8)
ax.plot(np.log2(df_20["N"]), df_20["totalTime(ms)"], label="GPU Time 20-OP", marker='o', color=gpu_colors[2], linewidth=4, alpha=0.8)
ax.plot(np.log2(df_40["N"]), df_40["totalTime(ms)"], label="GPU Time 40-OP", marker='o', color=gpu_colors[3], linewidth=4, alpha=0.8)
ax.plot(np.log2(df_1["N"]), df_1["cpuTime(ms)"], label="CPU Time 1-OP", marker='o', color=cpu_color_1, linewidth=4, alpha=0.8)
ax.plot(np.log2(df_4["N"]), df_4["cpuTime(ms)"], label="CPU Time 4-OP", marker='o', color=cpu_color_4, linewidth=4, alpha=0.8)

ax.tick_params(axis="both", which="major", labelsize=14)
ax.set_xlabel("Size (N, log2 scale)", fontsize=14)
ax.set_ylabel("Time (ms)", fontsize=14)
ax.set_xlim(12, 24)
ax.set_ylim(0, 10)
ax.grid(True, which="both", ls="--", lw=2.0, alpha=0.5)
ax.legend(fontsize=14)

fig.suptitle("Total Time", fontsize=18)
fig.savefig("total_time.png")
plt.close()


fig, ax = plt.subplots(1, 1, dpi=180, layout='constrained', figsize=(15, 5))
ax.plot(np.log2(df_20["N"]), df_20["allocateTime(ms)"], label="Allocate Time", marker='o', color='blue', linewidth=2)
ax.plot(np.log2(df_20["N"]), df_20["loadTime(ms)"], label="Load Time", marker='o', color='orange', linewidth=2)
ax.plot(np.log2(df_20["N"]), df_20["calcTime(ms)"], label="Calc Time", marker='o', color='green', linewidth=2)
ax.plot(np.log2(df_20["N"]), df_20["loadTimeBack(ms)"], label="Load Back Time", marker='o', color='red', linewidth=2)
ax.set_xlabel("Size (N, log2 scale)")
ax.set_ylabel("Time (ms)")
ax.set_xlim(25, 30)
ax.set_ylim(-100, 500)
ax.grid(True, which="both", ls="--", lw=0.5)
ax.legend()
fig.suptitle("Breakdown of GPU Time", fontsize=14, fontweight="bold")
fig.savefig("gpu_time_breakdown.png")
plt.close()
