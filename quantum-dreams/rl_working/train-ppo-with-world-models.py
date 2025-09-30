#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ====== 依赖 ======
import os, glob, time, pickle, argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import wandb

# ====== 你已有的环境封装 ======
import legacy_jax_compat  # noqa
from jax.lib import xla_bridge
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
import sys
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))

from rl_working.envs.utils.wrappers import VecEnv, LogWrapper
from rl_working.envs.single_rydberg_env import RydbergEnv
from rl_working.envs.single_stirap_env import SimpleStirap
from rl_working.envs.multistep_stirap_env import MultiStirap
from rl_working.envs.single_rydberg_two_photon_env import RydbergTwoEnv
from rl_working.envs.single_transmon_reset_env import TransmonResetEnv
from rl_working.env_configs.configs import (
    get_plot_elem_names, get_simple_stirap_params, get_multi_stirap_params,
    get_rydberg_cz_params, get_rydberg_two_params, get_transmon_reset_params
)
import matplotlib.pyplot as plt
# —— 最原始落盘：一行两个数 —— 
SIMPLE_TXT = "episodes_simple.txt"
# 可选：写表头（不需要也行）
if not os.path.exists(SIMPLE_TXT):
    with open(SIMPLE_TXT, "w") as _f:
        _f.write("# timestep  returns_mean  fid_mean\n")

# ====== 新增：结果落盘目录与 CSV（不影响原逻辑） ======
RESULT_DIR = "results_metrics"
os.makedirs(RESULT_DIR, exist_ok=True)
EPISODE_CSV = os.path.join(RESULT_DIR, "episodes.csv")
if not os.path.exists(EPISODE_CSV):
    with open(EPISODE_CSV, "w") as f:
        f.write("timestep,env_idx,episode_reward,episode_fid\n")

def _append_episode_metrics(timestep:int, env_idx:int, reward:float, fid_val):
    """将单条 episode 指标追加到 CSV。fid_val 允许为 None。"""
    with open(EPISODE_CSV, "a") as f:
        if fid_val is None:
            f.write(f"{int(timestep)},{int(env_idx)},{float(reward)},\n")
        else:
            f.write(f"{int(timestep)},{int(env_idx)},{float(reward)},{float(fid_val)}\n")

# ====== 基本信息 ======
print("Platform used: ")
print(xla_bridge.get_backend().platform)
print(jax.devices())
jax.config.update("jax_default_device", jax.devices()[0])
print("JAX backend: ", jax.default_backend())

# GPU/CPU 浮点精度
if "cuda" in str(jax.devices()):
    print("Connected to a GPU")
    default_dtype = jnp.float32
else:
    jax.config.update("jax_platform_name", "cpu")
    print("Not connected to a GPU")
    jax.config.update("jax_enable_x64", True)
    default_dtype = jnp.float64


# ====== 工具：保存 PPO 回放（原样保留） ======
def save_rollout_npz(rollout_batch, run_id="default", global_step=0, outdir="ppo_rollouts"):
    os.makedirs(outdir, exist_ok=True)
    ts = str(time.time()).replace(".", "")
    fpath = os.path.join(outdir, f"rollout_T{rollout_batch['obs'].shape[0]}_"
                                    f"step{global_step}_{run_id}_{ts}.npz")
    np.savez_compressed(
        fpath,
        obs=np.array(rollout_batch["obs"]),
        actions=np.array(rollout_batch["actions"]),
        rewards=np.array(rollout_batch["rewards"]),
        dones=np.array(rollout_batch["dones"]),
    )
    print(f"[saved] PPO rollout to {fpath}")
    return fpath


# ====== 世界模型：只要 LSTM 隐状态 h（冻结，不训练） ======
class WM_LSTM(nn.Module):
    """纯 JAX LSTM：给 (obs, prev_action) -> h_t"""
    hidden_size: int
    in_dim: int  # = obs_dim + act_dim

    @nn.compact
    def __call__(self, x, *, initial_state=None):
        """
        x: [T, B, in_dim]  (这里我们每次喂 T=1)
        return: hs [T, B, H], (hT, cT)
        """
        T, B, in_dim = x.shape
        H = self.hidden_size

        # —— 关键修复：参数 dtype 与输入 x.dtype 一致 ——
        k_init = nn.initializers.lecun_normal()
        W_x = self.param("W_x", lambda k, s: k_init(k, s, x.dtype), (in_dim, 4 * H))
        W_h = self.param("W_h", lambda k, s: k_init(k, s, x.dtype), (H, 4 * H))
        b   = self.param("b",   lambda k, s: nn.initializers.zeros(k, s, x.dtype), (4 * H,))

        if initial_state is None:
            h0 = jnp.zeros((B, H), dtype=x.dtype)
            c0 = jnp.zeros((B, H), dtype=x.dtype)
        else:
            h0, c0 = initial_state

        def step(carry, xt):
            h, c = carry
            gates = xt @ W_x + h @ W_h + b
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            i = jax.nn.sigmoid(i)
            f = jax.nn.sigmoid(f)
            o = jax.nn.sigmoid(o)
            g = jnp.tanh(g)
            c_new = f * c + i * g
            h_new = o * jnp.tanh(c_new)
            return (h_new, c_new), h_new

        (hT, cT), hs = jax.lax.scan(step, (h0, c0), x)
        return hs, (hT, cT)


def wm_init_params(rng, obs_dim, act_dim, hidden=256):
    """初始化 WM 参数；如果你有已训练好的 mdrnn_flax.msgpack，可以在此加载替换。"""
    wm = WM_LSTM(hidden_size=hidden, in_dim=obs_dim + act_dim)
    # —— 关键修复：dummy 的 dtype 使用全局默认（CPU=fp64/GPU=fp32） ——
    dummy = jnp.zeros((1, 1, obs_dim + act_dim), dtype=default_dtype)
    params = wm.init(rng, dummy)["params"]
    return wm, params

# —— 关键修复：carry 支持传入 dtype，与观测一致 ——
def wm_reset_carry(batch_size, hidden, dtype):
    h0 = jnp.zeros((batch_size, hidden), dtype=dtype)
    c0 = jnp.zeros((batch_size, hidden), dtype=dtype)
    return (h0, c0)

def wm_step(wm_apply, wm_params, carry, obs, prev_action):
    """
    obs: [B, obs_dim], prev_action: [B, act_dim]
    return: h: [B, H], new_carry
    """
    x = jnp.concatenate([obs, prev_action], axis=-1)[None, ...]  # [1,B,in_dim]
    hs, new_carry = wm_apply({"params": wm_params}, x, initial_state=carry)
    h = hs[-1]  # [B,H]
    return h, new_carry


# ====== PPO 网络（原样保留，只是输入维度会变大 = obs_dim + H） ======
class CombinedActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"
    layer_size: int = 128

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "elu":
            activation = nn.elu
        elif self.activation == "leaky_relu":
            activation = nn.leaky_relu
        elif self.activation == "relu6":
            activation = nn.relu6
        elif self.activation == "selu":
            activation = nn.selu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(self.layer_size,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.layer_size,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean_val = nn.Dense(self.action_dim,
                                  kernel_init=orthogonal(0.01),
                                  bias_init=constant(0.0))(actor_mean)

        actor_logtstd = self.param("log_std", nn.initializers.zeros,
                                   (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean_val,
                                            jnp.exp(actor_logtstd))

        critic = nn.Dense(1,
                          kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(actor_mean)

        return pi, jnp.squeeze(critic, axis=-1)


# ====== 轨迹结构（原样） ======
from typing import NamedTuple
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ====== 可选的几个环境 ======
envs_class_dict = {
    "simple_stirap": SimpleStirap,
    "multi_stirap":  MultiStirap,
    "rydberg":       RydbergEnv,
    "rydberg_two":   RydbergTwoEnv,
    "transmon_reset": TransmonResetEnv,
}


# ====== PPO 训练工厂：仅在喂网络前插入 WM 隐状态拼接 ======
def PPO_make_train(config):
    env = envs_class_dict[config["ENV_NAME"]](**config["ENV_PARAMS"])
    env = LogWrapper(env)
    env = VecEnv(env)
    env_params = env.default_params

    def linear_schedule(count):
        frac = (1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) /
                config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(rng):
        # ---- 初始化世界模型 ----
        obs_dim = env.observation_space(env_params).shape[0]
        act_dim = env.action_space(env_params).shape[0]
        WM_H = config.get("WM_H", 256)

        rng, _rng = jax.random.split(rng)
        wm_module, wm_params = wm_init_params(_rng, obs_dim, act_dim, hidden=WM_H)
        wm_apply = wm_module.apply  # 供后续调用

        # ---- 初始化 PPO 网络：输入维 = obs_dim + WM_H ----
        network = CombinedActorCritic(
            action_dim=act_dim,
            activation=config["ACTIVATION"],
            layer_size=config["LAYER_SIZE"],
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((obs_dim + WM_H,), dtype=jnp.float32)   # 关键：拼接后的维度
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # ---- 初始化环境、WM carry、prev_action ----
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)  # obsv: [B, obs_dim]

        # —— 关键修复：carry/prev_action 使用与观测相同 dtype ——
        wm_carry = wm_reset_carry(config["NUM_ENVS"], WM_H, obsv.dtype)   # (h,c)
        prev_action = jnp.zeros((config["NUM_ENVS"], act_dim), dtype=obsv.dtype)

        start_time = time.time()
        step = 0

        # ---- 训练循环 ----
        def _update_step(runner_state, unused):
            # -------- 采样 --------
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, step, rng, wm_params, wm_carry, prev_action = runner_state

                # 世界模型：基于 (obs, prev_action) 得到 h_t，并与 obs 拼接
                h, wm_carry = wm_step(wm_apply, wm_params, wm_carry, last_obs, prev_action)
                combined_obs = jnp.concatenate([last_obs, h], axis=-1)

                # 选动作
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, combined_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # 环境前进一步
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

                transition = Transition(done, action, value, reward, log_prob, last_obs, info)

                # 更新 prev_action
                prev_action = action

                runner_state = (train_state, env_state, obsv, step, rng, wm_params, wm_carry, prev_action)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # 保存回放（原样）
            if config.get("SAVE_WM_DATA", True):
                rollout_batch = {
                    "obs":     traj_batch.obs,
                    "actions": traj_batch.action,
                    "rewards": traj_batch.reward,
                    "dones":   traj_batch.done,
                }
                run_id = wandb.run.id if (config.get("LOG_WAND") and wandb.run) else "no_wandb"
                global_step = runner_state[3]

                def _save_cb(batch, _run_id, _gstep):
                    import numpy as _np
                    batch_np = {
                        "obs":     _np.array(batch["obs"]),
                        "actions": _np.array(batch["actions"]),
                        "rewards": _np.array(batch["rewards"]),
                        "dones":   _np.array(batch["dones"]),
                    }
                    save_rollout_npz(batch_np, run_id=_run_id, global_step=int(_gstep))
                    return 0
                jax.debug.callback(_save_cb, rollout_batch, run_id, global_step)

            # 强制 episode 结束后 reset（与你原逻辑一致）
            train_state, env_state, obsv, step, rng, wm_params, wm_carry, prev_action = runner_state
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = env.reset(reset_rng, env_params)
            # reset 后 prev_action 清零（保持 dtype）
            prev_action = jnp.zeros_like(prev_action, dtype=obsv.dtype)

            step = step + 1
            runner_state = (train_state, env_state, obsv, step, rng, wm_params, wm_carry, prev_action)

            # -------- 计算 GAE --------
            train_state, env_state, last_obs, step, rng, wm_params, wm_carry, prev_action = runner_state

            # 末值估计时同样用 WM 拼接
            h_last, wm_carry = wm_step(wm_apply, wm_params, wm_carry, last_obs, prev_action)
            combined_last = jnp.concatenate([last_obs, h_last], axis=-1)
            _, last_val = network.apply(train_state.params, combined_last)
            step = 0

            def _calculate_gae(traj_batch, last_val):
                last_val = last_val.astype(default_dtype)
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae)
                    gae = gae.astype(default_dtype)
                    value = value.astype(default_dtype)
                    return (gae, value), gae
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val, dtype=default_dtype), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # -------- 更新网络（原逻辑不变） --------
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    # 这里在更新阶段要重新跑前向：同样在 obs 前拼接 h（用 wm 的“离线重放”）
                    def _loss_fn(params, traj_batch, gae, targets):
                        # 复原 prev_action 序列：首帧补零，之后等于上一拍动作
                        if traj_batch.action.ndim == 3:   # [T,B,A]
                            B = traj_batch.action.shape[1]
                            A = traj_batch.action.shape[2]
                            zero_first = jnp.zeros((1, B, A), dtype=traj_batch.action.dtype)
                            prev_actions_seq = jnp.concatenate([zero_first, traj_batch.action[:-1]], axis=0)  # [T,B,A]
                        else:  # [B,A]
                            prev_actions_seq = jnp.zeros_like(traj_batch.action)

                        def roll_wm(carry, ta):
                            _obs, _prev_act = ta
                            h, carry2 = wm_step(wm_apply, wm_params, carry, _obs, _prev_act)
                            x = jnp.concatenate([_obs, h], axis=-1)
                            return carry2, x

                        if traj_batch.obs.ndim == 3:
                            carry0 = wm_reset_carry(traj_batch.obs.shape[1], WM_H, traj_batch.obs.dtype)
                            _, combined_seq = jax.lax.scan(roll_wm, carry0, (traj_batch.obs, prev_actions_seq))
                            x_in = combined_seq  # [T,B,obs+H]
                        else:
                            # 小批是 [N, obs_dim]，这里一次性对 N 个样本算 WM hidden
                            B = traj_batch.obs.shape[0]  # N
                            carry0 = wm_reset_carry(B, WM_H, traj_batch.obs.dtype)
                            # prev_actions_seq 在上面已构造为与 obs 同形状（[N, A]）
                            h, _ = wm_step(wm_apply, wm_params, carry0, traj_batch.obs, prev_actions_seq)  # [N, H]
                            x_in = jnp.concatenate([traj_batch.obs, h], axis=-1)  # [N, obs_dim+H]


                        pi, value = network.apply(params, x_in)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, aux)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, loss_pack = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_pack

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_pack = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            step = runner_state[3]

            # ---- 日志（原样） ----
            if config.get("LOGGING"):
                def callback(infos):
                    info, loss_pack, step = infos
                    timestep = config["NUM_ENVS"] * config["NUM_STEPS"] * step
                    if step % config["LOG_FREQ"] != 0:
                        return
                    min_fidelity = jnp.min(info["fid"]); max_fidelity = jnp.max(info["fid"]); std_fidelity = jnp.std(info["fid"])
                    speed_1k = (time.time() - start_time) * 1e3 / max(timestep,1)
                    print(f"time per 1k steps: {speed_1k} seconds")

                    wandb_log_dict = {
                        "timestep": timestep,
                        "time_per_1k_steps": speed_1k,
                        "max_fidelity": max_fidelity,
                        "min_fidelity": min_fidelity,
                        "std_fidelity": std_fidelity,
                        "total_loss": jnp.mean(jnp.ravel(loss_pack[0])),
                        "value_loss": jnp.mean(jnp.ravel(loss_pack[1][0])),
                        "actor_loss": jnp.mean(jnp.ravel(loss_pack[1][1])),
                        "entropy": jnp.mean(jnp.ravel(loss_pack[1][2])),
                    }
                    if (wandb.run and timestep % (config["NUM_ENVS"] * config["LOG_FREQ"]) == 0):
                        elem_names = get_plot_elem_names(config["ENV_NAME"])
                        n_elem_names = len(elem_names)
                        fig, ax = plt.subplots(1, n_elem_names, figsize=(3 * n_elem_names, 3))
                        saved_elem = []
                        for elem_i, elem_name in enumerate(elem_names):
                            best_elem = info[elem_name][info["fid"] == max_fidelity][0]
                            saved_elem.append(best_elem)
                            x_values = np.linspace(0, 1, len(best_elem))
                            ax[elem_i].plot(x_values, best_elem)
                            ax[elem_i].set_title(f"{elem_name} vs Time")
                        timestr = str(time.time()).replace(".", "")
                        output_dir_img = "output_images_temp"
                        os.makedirs(output_dir_img, exist_ok=True)
                        fpath = os.path.join(output_dir_img, f"{timestr}.png")
                        plt.savefig(fpath); plt.close()
                        if config.get("LOG_WAND") and wandb.run:
                            wandb_log_dict["action_fig"] = wandb.Image(fpath)

                        env_name_var = config["ENV_NAME"]
                        directory = f"saved_data/{env_name_var}/{wandb.run.id if wandb.run else 'no_wb'}"
                        os.makedirs(directory, exist_ok=True)
                        data_fpath = os.path.join(directory, f"{env_name_var}_{int(timestep)}.pkl")
                        with open(data_fpath, "wb") as file:
                            pickle.dump(saved_elem, file)

                    for log_elem in info.keys():
                        if "returned_episode" not in log_elem:
                            continue
                        return_values = info[log_elem][info["returned_episode"]]
                        log_val_name = log_elem.split("_")[-1]
                        mean_value = np.mean(return_values)
                        print(f"global step={timestep}, episodic {log_val_name} mean={mean_value}")
                        wandb_log_dict = {f"episodic_{log_val_name}_mean": mean_value}
                        if config.get("LOG_WAND") and wandb.run:
                            wandb.log(wandb_log_dict)

# === 最原始：把两个小数记到一个 txt 里（每行：timestep returns_mean fid_mean） ===
                        try:
                            mask = np.array(info.get("returned_episode", []), dtype=bool)
                            if mask.size and mask.any():
                                # 回合“总回报”的键：优先 returns，其次 rewards（不同环境命名不同）
                                ret_mean = None
                                if "returned_episode_returns" in info:
                                    vals = np.array(info["returned_episode_returns"])[mask]
                                    if vals.size: ret_mean = float(vals.mean())
                                elif "returned_episode_rewards" in info:
                                    vals = np.array(info["returned_episode_rewards"])[mask]
                                    if vals.size: ret_mean = float(vals.mean())
                                fid_mean = None
                                if "fid" in info:
                                    fvals = np.array(info["fid"])[mask]
                                    if fvals.size: fid_mean = float(fvals.mean())

                                r = 0.0 if ret_mean is None else ret_mean
                                f = 0.0 if fid_mean is None else fid_mean
                                with open(SIMPLE_TXT, "a") as _f:
                                    _f.write(f"{int(timestep)} {r:.6f} {f:.6f}\n")
                        except Exception as e:
                            print("[simple save failed]", e)


                jax.debug.callback(callback, (metric, loss_pack, step))

            runner_state = (train_state, env_state, obsv, step, rng, wm_params, wm_carry, prev_action)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, step, _rng, wm_params, wm_carry, prev_action)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state}

    return train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_updates", type=int, default=5000)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--layer_size", type=int, default=256)
    parser.add_argument("--anneal_lr", type=int, default=0)

    parser.add_argument("--env",
        choices=["multi_stirap","simple_stirap","rydberg","rydberg_two","transmon_reset"],
        default="simple_stirap")

    # 噪声与各环境参数（原样）
    parser.add_argument("--noise", choices=["None", "ou", "g"], default="None")
    parser.add_argument("--sigma_phase", type=float, default=0)
    parser.add_argument("--sigma_amp", type=float, default=0)
    parser.add_argument("--mu_phase", type=float, default=0)
    parser.add_argument("--mu_amp", type=float, default=0)
    parser.add_argument("--gamma_ss", type=float, default=1)
    parser.add_argument("--omega_ss", type=float, default=30)
    parser.add_argument("--delta_ss", type=float, default=20)
    parser.add_argument("--x_detuning_ss", type=float, default=100)
    parser.add_argument("--final_state_zero_ss", type=float, default=0.0)
    parser.add_argument("--area_pen_ss", type=float, default=0.0)
    parser.add_argument("--smoothness_pen_ss", type=float, default=0.001)
    parser.add_argument("--smoothness_pen_ss_det", type=float, default=0.001)
    parser.add_argument("--fix_endpoints_ss", type=int, default=1)
    parser.add_argument("--smoothness_calc_amp", type=str, default="second_derivative")
    parser.add_argument("--smoothness_calc_det", type=str, default="second_derivative")
    parser.add_argument("--smoothness_cutoff_freq", type=float, default=5.0)
    parser.add_argument("--log_fidelity", type=int, default=1)
    parser.add_argument("--kernel_std_amp", type=float, default=4.0)
    parser.add_argument("--kernel_std_freq", type=float, default=4.0)

    parser.add_argument("--blockade_strength", type=float, default=500)
    parser.add_argument("--const_freq_pump_rydberg_two", type=int, default=0)
    parser.add_argument("--const_amp_stokes_rydberg_two", type=int, default=0)

    parser.add_argument("--n_sections_multi", type=int, default=1)
    parser.add_argument("--multi_use_beta", type=int, default=0)
    parser.add_argument("--mxstep_solver", type=int, default=1000)
    parser.add_argument("--mx_step_penalty", type=float, default=-10.0)

    # 世界模型隐藏维（可改）
    parser.add_argument("--wm_hidden", type=int, default=256)

    args = parser.parse_args()
    assert args.num_envs % args.num_minibatches == 0

    # 组装配置
    config = {
        "LR": args.lr,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": 1,
        "NUM_UPDATES": args.num_updates,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": args.num_minibatches,
        "MINIBATCH_SIZE": args.num_envs // args.num_minibatches,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "ACTIVATION": "relu6",
        "LAYER_SIZE": args.layer_size,
        "ENV_NAME": args.env,
        "ANNEAL_LR": args.anneal_lr,
        "DEBUG": True, "DEBUG_NOJIT": False,
        "LOGGING": True, "LOG_FREQ": 1, "LOG_WAND": True,
        "LOCAL_LOGGING": True, "LOCAL_SAVE_NAME": "local_save",
        "MU_PHASE": args.mu_phase, "MU_AMP": args.mu_amp,
        "ALPHA_PHASE": 0.1, "ALPHA_AMP": 0.1,
        "SIGMA_PHASE": args.sigma_phase, "SIGMA_AMP": args.sigma_amp,
        "WM_H": args.wm_hidden,
    }

    # 环境参数（原样）
    if config["ENV_NAME"] == "simple_stirap":
        config["ENV_PARAMS"] = get_simple_stirap_params(args)
    elif config["ENV_NAME"] == "rydberg":
        config["ENV_PARAMS"] = get_rydberg_cz_params(args); config["LR"] = 8e-4
    elif config["ENV_NAME"] == "multi_stirap":
        config["NUM_STEPS"] = args.n_sections_multi
        config["MINIBATCH_SIZE"] = config["MINIBATCH_SIZE"] * config["NUM_STEPS"]
        config["ENV_PARAMS"] = get_multi_stirap_params(args)
        config["ENV_PARAMS"]["n_sections"] = args.n_sections_multi
        config["ENV_PARAMS"]["n_action_steps"] = 50
        config["ENV_PARAMS"]["use_mu_beta"] = args.multi_use_beta
    elif config["ENV_NAME"] == "rydberg_two":
        config["ENV_PARAMS"] = get_rydberg_two_params(args)
    elif config["ENV_NAME"] == "transmon_reset":
        config["ENV_PARAMS"] = get_transmon_reset_params(args)
    else:
        raise ValueError("Environment not recognized")

    config["ENV_PARAMS"]["ou_noise_params"] = [
        config["MU_PHASE"], config["MU_AMP"], config["ALPHA_PHASE"], config["ALPHA_AMP"],
        config["SIGMA_PHASE"], config["SIGMA_AMP"],
    ]

    if config["DEBUG_NOJIT"]:
        jax.disable_jit(disable=True)

    config["NUM_ENVS"] = args.num_envs
    assert (config["NUM_MINIBATCHES"] * config["MINIBATCH_SIZE"]
            == config["NUM_STEPS"] * config["NUM_ENVS"])

    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)

    single_train = jit(PPO_make_train(config))
    print(f"Starting a Run of {config['NUM_UPDATES']} Updates")

    if config["LOG_WAND"]:
        wandb.init(project="", entity="", config=config)

    outs = jax.block_until_ready(single_train(rng))
