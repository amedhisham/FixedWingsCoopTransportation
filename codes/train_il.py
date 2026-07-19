"""
train_il.py — supervised imitation of the optimizer's lambda (Formulation 1).

Loads il_dataset.npz, standardizes observations, and trains Actor.forward (the
mean) to regress lambda by MSE, on GPU.

The experiment: compare the achieved MSE to the predict-the-mean baseline,
Var(lambda).
  - TRAIN MSE ~ Var(lambda)   -> lambda is NOT a function of obs (phase not
                                 observable); IL-of-lambda is ill-posed.
  - TRAIN MSE -> ~0           -> lambda IS recoverable from obs (phase encoded
                                 in the drone's orbital state).
  - VAL MSE low too           -> it also generalizes (within this trajectory).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from networks import Actor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 300
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.2
SEED = 0


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    data = np.load("il_dataset.npz")
    obs = data["obs"].astype(np.float32)          # (M, 24)
    lam = data["lam"].astype(np.float32)          # (M, 1)
    M = len(obs)

    # Standardize observations (save stats for deployment).
    obs_mean = obs.mean(0, keepdims=True)
    obs_std = obs.std(0, keepdims=True) + 1e-6
    obs_n = (obs - obs_mean) / obs_std

    var_lambda = float(lam.var())                 # predict-the-mean baseline MSE

    # Random 80/20 split.
    idx = rng.permutation(M)
    n_val = int(VAL_FRAC * M)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    t = lambda a: torch.tensor(a, device=DEVICE)
    tr_obs, tr_lam = t(obs_n[tr_idx]), t(lam[tr_idx])
    va_obs, va_lam = t(obs_n[val_idx]), t(lam[val_idx])
    loader = DataLoader(TensorDataset(tr_obs, tr_lam), batch_size=BATCH, shuffle=True)

    net = Actor(obs_dim=obs.shape[1], act_dim=1).to(DEVICE)   # obs_dim follows the dataset (now 38 with clock)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    print(f"device={DEVICE}  samples={M}  Var(lambda)={var_lambda:.4f} (baseline MSE)\n")
    tr_hist, va_hist = [], []
    best_va, best_ep = float("inf"), 0
    for ep in range(1, EPOCHS + 1):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            mse(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tr = mse(net(tr_obs), tr_lam).item()
            va = mse(net(va_obs), va_lam).item()
        tr_hist.append(tr); va_hist.append(va)
        if va < best_va:                          # checkpoint the BEST-val model only
            best_va, best_ep = va, ep
            torch.save({"state_dict": net.state_dict(),
                        "obs_mean": obs_mean, "obs_std": obs_std}, "il_actor.pt")
        print(f"ep {ep:3d}   train MSE {tr:.4f}   val MSE {va:.4f}")

    print(f"\nsaved il_actor.pt  (best val MSE {best_va:.4f} at epoch {best_ep})")

    # Training curve.
    epochs = range(1, EPOCHS + 1)
    plt.figure()
    plt.plot(epochs, tr_hist, label="train MSE")
    plt.plot(epochs, va_hist, label="val MSE")
    plt.axhline(var_lambda, ls="--", c="gray", label="baseline = Var(lambda)")
    plt.axvline(best_ep, ls=":", c="green", label=f"best val (ep {best_ep})")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("IL training curve (lambda regression)")
    plt.legend(); plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
