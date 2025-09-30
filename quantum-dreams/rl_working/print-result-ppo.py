import numpy as np
import matplotlib.pyplot as plt

# 文件路径
fname = "reward_fid_per_update.txt"

# 读取数据（跳过表头）
data = np.loadtxt(fname, comments="#")

# 提取三列
timestep = data[:, 0]
returns = data[:, 1]
fid = data[:, 2]

# === 排序 ===
sort_idx = np.argsort(timestep)
timestep = timestep[sort_idx]
returns = returns[sort_idx]
fid = fid[sort_idx]

# === 平滑函数 ===
def moving_avg(x, k=7):
    if k <= 1:
        return x
    k = min(len(x), k)
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

SMOOTH_WIN = 1  # 窗口大小
TAIL_RAW   = 1   # 最后多少个点保持原始值

returns_s = moving_avg(returns, SMOOTH_WIN)
fid_s     = moving_avg(fid, SMOOTH_WIN)

# 覆盖尾部，避免掉落
returns_s[-TAIL_RAW:] = returns[-TAIL_RAW:]
fid_s[-TAIL_RAW:]     = fid[-TAIL_RAW:]

# === 画图 ===
plt.figure(figsize=(8, 5))
plt.plot(timestep*160, returns_s*10, label=f"returns_mean (smooth {SMOOTH_WIN})", marker="o", markersize=3)
plt.plot(timestep*160, fid_s*10, label=f"fid_mean (smooth {SMOOTH_WIN})", marker="x", markersize=3)

plt.xlabel("Timestep")
plt.ylabel("Value")
plt.title("Returns and Fidelity vs Timestep (traditional-ppo)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("reward_fid_plot_smooth.png", dpi=200)
plt.close()

print("[saved] -> reward_fid_plot_smooth.png")
