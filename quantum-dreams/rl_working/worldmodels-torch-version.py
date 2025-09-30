#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal

# ---------------- Dataset (robust) ----------------
class ReplayDatasetNPZ(Dataset):
    def __init__(self, folder, target_shape=None):
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

        # 自动选择目标维度：出现次数最多的
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
        obs = torch.tensor(arr["obs"], dtype=torch.float32)       # [T,B,obs_dim]
        actions = torch.tensor(arr["actions"], dtype=torch.float32)
        rewards = torch.tensor(arr["rewards"], dtype=torch.float32)
        dones = torch.tensor(arr["dones"], dtype=torch.float32)
        return obs, actions, rewards, dones

# ---------------- GMM Loss ----------------
def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    batch = batch.unsqueeze(-2)   # [T,B,1,latent]

   
    sigmas = sigmas.clamp(min=1e-3, max=1e3)

    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)    # [T,B,K,latent]
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)  # [T,B,K]
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs
    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)  # [T,B]

    log_prob = max_log_probs.squeeze(-1) + torch.log(probs + 1e-8)

    if reduce:
        return -torch.mean(log_prob)
    return -log_prob


# ---------------- MDRNN ----------------
class MDRNN(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        self.rnn = nn.LSTM(latents + actions, hiddens)
        self.gmm_linear = nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, actions, latents):
        seq_len, bs = actions.size(0), actions.size(1)
        ins = torch.cat([actions, latents], dim=-1)  # [T,B,lat+act]
        outs, _ = self.rnn(ins)   # [T,B,H]
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride].view(seq_len, bs, self.gaussians, self.latents)
        sigmas = gmm_outs[:, :, stride:2*stride].view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2*stride:2*stride+self.gaussians].view(seq_len, bs, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]
        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

# ---------------- Training ----------------
def train_mdrnn(data_folder, epochs=10, hiddens=256, gaussians=5, lr=1e-3):
    dataset = ReplayDatasetNPZ(data_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim
    print(f"[dim] obs_dim={obs_dim}, act_dim={act_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDRNN(obs_dim, act_dim, hiddens, gaussians).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for obs, act, rew, done in dataloader:
            obs, act, rew, done = obs.squeeze(0).to(device), act.squeeze(0).to(device), \
                                  rew.squeeze(0).to(device), done.squeeze(0).to(device)

            obs_in, act_in, obs_next = obs[:-1], act[:-1], obs[1:]
            rew_t, done_t = rew[:-1], done[:-1]

            mus, sigmas, logpi, rs, ds = model(act_in, obs_in)

            loss_gmm = gmm_loss(obs_next, mus, sigmas, logpi)
            loss_r = F.mse_loss(rs, rew_t)
            loss_d = F.binary_cross_entropy_with_logits(ds, done_t)

            loss = loss_gmm + loss_r + loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[epoch {epoch+1}] avg_loss={total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "mdrnn.pth")
    print("[saved] mdrnn.pth")

if __name__ == "__main__":
    train_mdrnn("ppo_rollouts", epochs=50, hiddens=256, gaussians=5, lr=1e-3)
