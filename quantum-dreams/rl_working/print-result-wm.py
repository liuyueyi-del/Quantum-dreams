#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

# === 配置 ===
fname = "episodes_simple.txt"
out_img = "reward_fid_curve.png"
SMOOTH_WIN = 1   # 平滑窗口大小（可调）

# === 读取数据 ===
data = np.loadtxt(fname)

# 按 timestep 去重（只保留每组的第一条）
timestep, first_idx = np.unique(data[:, 0], return_index=True)
reward = data[first_idx, 1]
fid    = data[first_idx, 2]

# === 滑动平均函数（保留最后几个原始值） ===
def moving_avg_tail(x, k=5):
    if k <= 1:
        return x
    k = min(k, len(x))
    w = np.ones(k, dtype=float) / k
    smooth = np.convolve(x, w, mode="same")
    # 保留最后 (k//2) 个点的原始值
    tail = k // 2
    if tail > 0:
        smooth[-tail:] = x[-tail:]
    return smooth

reward_s = moving_avg_tail(reward, SMOOTH_WIN)
fid_s    = moving_avg_tail(fid, SMOOTH_WIN)

# === 画图 ===
plt.figure(figsize=(8, 5))
plt.plot(timestep, reward_s, label="returns_mean (smooth+tail)", marker="o", ms=3, lw=1.6)
plt.plot(timestep, fid_s, label="fid_mean (smooth+tail)", marker="x", ms=3, lw=1.6, linestyle="--")

plt.xlabel("Timestep")
plt.ylabel("Value")
plt.title(f"Returns and Fidelity vs Timestep (our world-models method)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像
plt.savefig(out_img, dpi=200)
plt.close()

print(f"[OK] 曲线已保存到: {out_img}")
