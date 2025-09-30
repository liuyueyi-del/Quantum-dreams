#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JAX/Flax 版本的 MDRNN 训练脚本（纯 JAX LSTM，避免 tracer 泄漏）
- 数据：每个 .npz 文件包含 obs, actions, rewards, dones，形状 [T,B,D]
- 模型：LSTM -> Dense 输出 GMM 参数 + 奖励 + done logits
- 损失：GMM NLL（预测 obs_{t+1}）、奖励 MSE、done 的 BCE-with-logits
- 训练：逐文件（等价 batch_size=1），沿时间维用 lax.scan
- 保存：mdrnn_flax.msgpack
"""

import os, glob
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from flax import linen as nn
from flax.training import train_state
from flax.serialization import to_bytes
import optax


# ---------------- Dataset (robust) ----------------
class ReplayDatasetNPZ:
    def __init__(self, folder, target_shape: Tuple[int, int] = None):
        files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if len(files) == 0:
            raise RuntimeError(f"No .npz files found in {folder}")

        shape_counts = {}
        file_shapes = []
        for fp in files:
            try:
                arr = np.load(fp)
                obsd = int(arr["obs"].shape[-1])
                actd = int(arr["actions"].shape[-1])
                key = (obsd, actd)
                shape_counts[key] = shape_counts.get(key, 0) + 1
                file_shapes.append((fp, key))
            except Exception as e:
                print(f"[skip] {fp} load failed: {e}")

        if not shape_counts:
            raise RuntimeError("No valid npz files with 'obs' and 'actions' found.")

        if target_shape is None:
            target_shape = max(shape_counts.items(), key=lambda x: x[1])[0]

        self.files = [fp for fp, key in file_shapes if key == target_shape]
        self.obs_dim, self.act_dim = target_shape

        print(f"[dataset] Found {len(files)} files in total")
        print(f"[dataset] Using shape (obs_dim={self.obs_dim}, act_dim={self.act_dim}) "
              f"with {len(self.files)} valid files")
        if len(self.files) == 0:
            raise RuntimeError("All files were filtered out due to shape mismatch.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        obs = arr["obs"].astype(np.float32)        # [T,B,obs_dim]
        actions = arr["actions"].astype(np.float32)
        rewards = arr["rewards"].astype(np.float32)
        dones = arr["dones"].astype(np.float32)
        return obs, actions, rewards, dones


# ---------------- LSTM Block (pure JAX in lax.scan) ----------------
class LSTMBlock(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x, *, initial_state=None):
        """
        纯 JAX 手写 LSTM（不在 scan 里调用任何 Flax 模块）
        x: [T, B, in_dim]
        return: hs [T, B, hidden_size], (hT, cT)
        """
        T, B, in_dim = x.shape
        H = self.hidden_size

        # 权重参数（一次性创建，在 scan 内只用 jnp 计算）
        k_init = nn.initializers.lecun_normal()
        W_x = self.param("W_x", k_init, (in_dim, 4 * H))         # 输入到门
        W_h = self.param("W_h", k_init, (H, 4 * H))              # 隐状态到门
        b   = self.param("b",   nn.initializers.zeros, (4 * H,)) # 偏置

        if initial_state is None:
            h0 = jnp.zeros((B, H), dtype=x.dtype)
            c0 = jnp.zeros((B, H), dtype=x.dtype)
        else:
            h0, c0 = initial_state

        def step(carry, xt):
            h, c = carry                      # [B,H], [B,H]
            gates = xt @ W_x + h @ W_h + b    # [B,4H]
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            i = jax.nn.sigmoid(i)
            f = jax.nn.sigmoid(f)
            o = jax.nn.sigmoid(o)
            g = jnp.tanh(g)
            c_new = f * c + i * g
            h_new = o * jnp.tanh(c_new)
            return (h_new, c_new), h_new      # y = h_new

        (hT, cT), hs = jax.lax.scan(step, (h0, c0), x)  # hs: [T,B,H]
        return hs, (hT, cT)


# ---------------- MDRNN ----------------
class MDRNN(nn.Module):
    latents: int
    actions: int
    hiddens: int
    gaussians: int

    @nn.compact
    def __call__(self, actions, latents):
        """
        actions: [T, B, action_dim]
        latents: [T, B, latent_dim]
        输出:
          mus:    [T,B,K,latent]
          sigmas: [T,B,K,latent] (正数)
          logpi:  [T,B,K] (对数分布)
          rs:     [T,B]
          ds:     [T,B] (logits)
        """
        x = jnp.concatenate([actions, latents], axis=-1)   # [T,B,lat+act]
        hs, _ = LSTMBlock(self.hiddens)(x)                 # [T,B,H]

        out_dim = (2 * self.latents + 1) * self.gaussians + 2
        gmm_outs = nn.Dense(out_dim, name="gmm_head")(hs)  # [T,B,out_dim]

        stride = self.gaussians * self.latents
        mus_flat   = gmm_outs[:, :, :stride]
        sig_flat   = gmm_outs[:, :, stride:2*stride]
        pi_flat    = gmm_outs[:, :, 2*stride:2*stride+self.gaussians]
        extras     = gmm_outs[:, :, 2*stride+self.gaussians:]  # [T,B,2]

        T, B = actions.shape[:2]
        mus    = mus_flat.reshape(T, B, self.gaussians, self.latents)
        sigmas = jnp.exp(sig_flat.reshape(T, B, self.gaussians, self.latents))
        sigmas = jnp.clip(sigmas, 1e-3, 1e3)

        logpi = jax.nn.log_softmax(pi_flat, axis=-1)  # [T,B,K]

        rs = extras[:, :, 0]   # [T,B]
        ds = extras[:, :, 1]   # [T,B] (logits)
        return mus, sigmas, logpi, rs, ds


# ---------------- Losses ----------------
def gmm_nll(obs_next, mus, sigmas, logpi):
    x = jnp.expand_dims(obs_next, axis=-2)  # [T,B,1,latent]
    const = jnp.log(2.0 * jnp.pi)
    z = (x - mus) / sigmas
    comp_log = -0.5 * (
        jnp.sum(z * z, axis=-1) +
        jnp.sum(2.0 * jnp.log(sigmas), axis=-1) +
        mus.shape[-1] * const
    )  # [T,B,K]
    g_log = logpi + comp_log
    max_log = jnp.max(g_log, axis=-1, keepdims=True)
    lse = max_log + jnp.log(jnp.sum(jnp.exp(g_log - max_log), axis=-1, keepdims=True) + 1e-8)
    log_prob = jnp.squeeze(lse, axis=-1)  # [T,B]
    return -jnp.mean(log_prob)


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def bce_with_logits(logits, labels):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels))


# ---------------- Train (use default TrainState) ----------------
TrainState = train_state.TrainState

def loss_fn(params, apply_fn, batch):
    obs, act, rew, done = batch
    if obs.shape[0] < 2:
        zero = jnp.array(0.0, dtype=obs.dtype)
        return zero, (zero, zero, zero)

    obs_in, act_in, obs_next = obs[:-1], act[:-1], obs[1:]
    rew_t, done_t = rew[:-1], done[:-1]

    mus, sigmas, logpi, rs, ds = apply_fn({"params": params}, act_in, obs_in)

    loss_gmm = gmm_nll(obs_next, mus, sigmas, logpi)
    loss_r   = mse_loss(rs, rew_t)
    loss_d   = bce_with_logits(ds, done_t)
    loss = loss_gmm + loss_r + loss_d
    return loss, (loss_gmm, loss_r, loss_d)


@jit
def train_step(state: TrainState, batch):
    def _lf(p):
        return loss_fn(p, state.apply_fn, batch)
    (loss, parts), grads = value_and_grad(_lf, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, parts


# ---------------- Training Loop ----------------
def train_mdrnn(data_folder, epochs=10, hiddens=256, gaussians=5, lr=1e-3, seed=0):
    dataset = ReplayDatasetNPZ(data_folder)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim
    print(f"[dim] obs_dim={obs_dim}, act_dim={act_dim}")

    # 初始化参数（用最小 dummy 输入确定 shape）
    model = MDRNN(latents=obs_dim, actions=act_dim, hiddens=hiddens, gaussians=gaussians)
    key = jax.random.PRNGKey(seed)
    dummy_act = jnp.zeros((1, 1, act_dim), dtype=jnp.float32)  # T-1=1
    dummy_lat = jnp.zeros((1, 1, obs_dim), dtype=jnp.float32)
    params = model.init(key, dummy_act, dummy_lat)["params"]

    tx = optax.adam(lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for epoch in range(epochs):
        total_loss = 0.0
        total_gmm = 0.0
        total_r   = 0.0
        total_d   = 0.0
        n_used = 0

        for i in range(len(dataset)):
            obs, act, rew, done = dataset[i]
            obs  = jnp.array(obs)
            act  = jnp.array(act)
            rew  = jnp.array(rew)
            done = jnp.array(done)

            if obs.shape[0] < 2:
                continue

            batch = (obs, act, rew, done)
            state, loss, parts = train_step(state, batch)
            total_loss += float(loss)
            gmm_l, r_l, d_l = [float(x) for x in parts]
            total_gmm += gmm_l
            total_r   += r_l
            total_d   += d_l
            n_used += 1

        n = max(1, n_used)
        print(f"[epoch {epoch+1}] "
              f"avg_loss={total_loss/n:.4f} | gmm={total_gmm/n:.4f} | r={total_r/n:.4f} | d={total_d/n:.4f}")

    out_path = "mdrnn_flax.msgpack"
    with open(out_path, "wb") as f:
        f.write(to_bytes(state.params))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    train_mdrnn("ppo_rollouts", epochs=50, hiddens=256, gaussians=5, lr=1e-3)
