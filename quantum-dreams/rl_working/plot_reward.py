#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot reward & fid curves from a whitespace- or comma-separated text file.
Expected columns: round, reward, fid
Header like "round  reward  fid" is fine (auto-skipped).

Examples:
  python plot_reward_fid.py \
      --file rl_working/per_action_logs/reward_fid_per_update.txt \
      --smooth 20 \
      --points 60 --resample bin \
      --out results/reward_fid_curve.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default="reward_fid_per_update.txt",
                   help="Path to the txt file with columns: round reward fid")
    p.add_argument("--out", type=str, default=None,
                   help="Output image path (png/pdf). Default: <file>.png next to the input")
    p.add_argument("--smooth", type=int, default=0,
                   help="Sliding window size for smoothing (0 = no smoothing)")
    p.add_argument("--points", type=int, default=0,
                   help="Resample to at most N points (0 = keep all)")
    p.add_argument("--resample", type=str, default="bin", choices=["bin","linspace"],
                   help="Resampling strategy when --points > 0: "
                        "'bin' uses equal-width bins and averages (smoother); "
                        "'linspace' picks indices evenly (no bin-average).")
    p.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    p.add_argument("--title", type=str, default="Reward & Fid vs Updates",
                   help="Figure title")
    return p.parse_args()

def _try_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _try_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def load_table(path):
    rounds, rewards, fids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue

            # split by whitespace or comma
            if "," in ln and " " not in ln:
                parts = [t.strip() for t in ln.split(",") if t.strip()]
            else:
                parts = [t for t in ln.replace(",", " ").split() if t]

            # Skip obvious header lines
            low = [t.lower() for t in parts]
            if any(k in low for k in ["round", "reward", "fid"]):
                continue
            if len(parts) < 3:
                continue

            r = _try_int(parts[0])
            rew = _try_float(parts[1])
            fid = _try_float(parts[2])
            if r is None or rew is None or fid is None:
                continue

            rounds.append(r)
            rewards.append(rew)
            fids.append(fid)

    if not rounds:
        raise RuntimeError(f"No valid data parsed from: {path}")
    # sort by round just in case
    idx = np.argsort(np.array(rounds))
    rounds = np.array(rounds)[idx]
    rewards = np.array(rewards)[idx]
    fids = np.array(fids)[idx]
    return rounds, rewards, fids

def smooth_curve(y, k):
    """Simple centered moving average (odd/even k both allowed)."""
    if k is None or k <= 1:
        return y
    k = int(k)
    k = max(1, min(k, len(y)))
    kernel = np.ones(k, dtype=float) / k
    pad = (k - 1) // 2
    ypad = np.pad(y, (pad, k - 1 - pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")

def resample_bin(x, ys, n_points):
    """Split x into n equal-width bins (by index) and average each bin."""
    if n_points <= 0 or len(x) <= n_points:
        return x, ys
    N = len(x)
    # bin edges in index space
    edges = np.linspace(0, N, n_points + 1, dtype=int)
    x_out, ys_out = [], [np.empty(n_points) for _ in ys]
    for i in range(n_points):
        sl = slice(edges[i], edges[i+1])
        if sl.start >= sl.stop:  # empty bin (shouldn't happen), skip
            continue
        x_out.append(x[sl].mean())
        for j, y in enumerate(ys):
            ys_out[j][i] = y[sl].mean()
    x_out = np.array(x_out)
    ys_out = [y for y in ys_out]
    return x_out, ys_out

def resample_linspace(x, ys, n_points):
    """Pick n_points indices evenly spaced (including first & last)."""
    if n_points <= 0 or len(x) <= n_points:
        return x, ys
    idx = np.linspace(0, len(x) - 1, n_points).round().astype(int)
    x_out = x[idx]
    ys_out = [y[idx] for y in ys]
    return x_out, ys_out

def main():
    args = parse_args()
    rounds, rewards, fids = load_table(args.file)

    # 1) optional smoothing
    if args.smooth and args.smooth > 1:
        rewards = smooth_curve(rewards, args.smooth)
        fids = smooth_curve(fids, args.smooth)
        # rounds 保持长度一致：用原 rounds 的等长版本
        # 上面的 smooth_curve 已保持长度不变，因此无需裁剪

    # 2) optional resampling to N points
    if args.points and args.points > 0:
        if args.resample == "bin":
            rounds, (rewards, fids) = resample_bin(rounds, [rewards, fids], args.points)
        else:  # linspace
            rounds, (rewards, fids) = resample_linspace(rounds, [rewards, fids], args.points)

    # 3) plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(rounds, rewards, label="reward", linewidth=2, marker="o", ms=3)
    l2, = ax2.plot(rounds, fids, label="fid", linewidth=2, linestyle="--", marker="x", ms=3)

    ax1.set_xlabel("update (round)")
    ax1.set_ylabel("reward")
    ax2.set_ylabel("fid")
    ax1.grid(True, alpha=0.25)
    fig.legend(handles=[l1, l2], loc="upper left", bbox_to_anchor=(0.12, 0.98))
    fig.suptitle(args.title)

    outpath = args.out
    if outpath is None:
        base, _ = os.path.splitext(args.file)
        outpath = base + ".png"

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    print(f"[saved] {outpath}")

if __name__ == "__main__":
    main()
