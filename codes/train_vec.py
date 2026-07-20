"""
train_vec.py — fit the whole-vector policy and settle the own-state question.

Fits two regressions on the optimizer data and compares R^2:
  A) whole-vector:   (load 18 + clock 14)            -> lambda[n]   (no carrier state)
  B) per-drone:      (load 18 + own 6 + clock 14)     -> lambda_i    (with own state)

If A's per-drone R^2 matches B's, own-state adds no predictive power and the
reconstruction-free whole-vector net is the one to deploy. Saves A -> il_actor_vec.pt.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from networks import Actor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 300
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.2
SEED = 0
N = 4


def fit(X, Y, label):
    """Standardize X, 80/20 split, train, return best net + stats + per-output val R^2."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    xm = X.mean(0, keepdims=True)
    xs = X.std(0, keepdims=True) + 1e-6
    Xn = ((X - xm) / xs).astype(np.float32)

    M = len(X)
    idx = rng.permutation(M)
    nv = int(VAL_FRAC * M)
    vi, ti = idx[:nv], idx[nv:]
    tt = lambda a: torch.tensor(a, device=DEVICE)
    tro, trl = tt(Xn[ti]), tt(Y[ti])
    vao, val = tt(Xn[vi]), tt(Y[vi])
    loader = DataLoader(TensorDataset(tro, trl), batch_size=BATCH, shuffle=True)

    net = Actor(obs_dim=X.shape[1], act_dim=Y.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); mse(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = mse(net(vao), val).item()
        if va < best:
            best = va
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pred = net(vao).cpu().numpy()
    yv = Y[vi]
    r2 = []
    for j in range(Y.shape[1]):
        ss_res = np.sum((yv[:, j] - pred[:, j]) ** 2)
        ss_tot = np.sum((yv[:, j] - yv[:, j].mean()) ** 2)
        r2.append(1.0 - ss_res / ss_tot)
    print(f"{label}\n   best val MSE {best:.4f}   R2 {[f'{r:.3f}' for r in r2]}")
    return net, xm.astype(np.float32), xs.astype(np.float32), r2


def main():
    d = np.load("vec_dataset.npz")
    load, clock, drone, lam = d["load"], d["clock"], d["drone"], d["lam"]   # lam (T, N)
    print(f"data: {len(load)} steps   Var(lambda)={lam.var():.4f}\n")

    # A) whole-vector: (load + clock) -> lambda[N]
    XA = np.concatenate([load, clock], axis=1)                       # (T, 32)
    netA, xmA, xsA, r2A = fit(XA, lam, "A  whole-vector  (load+clock -> lambda[4])")

    # B) per-drone: (load + own + clock) -> lambda_i, stacked over drones
    XB = np.concatenate([np.concatenate([load, drone[:, i, :], clock], axis=1) for i in range(N)], axis=0)
    YB = np.concatenate([lam[:, i:i + 1] for i in range(N)], axis=0)  # (N*T, 1)
    _, _, _, r2B = fit(XB, YB, "B  per-drone     (load+own+clock -> lambda)")

    print(f"\nR2  A whole-vector (no own state): mean {np.mean(r2A):.3f}")
    print(f"R2  B per-drone   (with own state): {r2B[0]:.3f}")
    print("-> if comparable, own state adds nothing; ship the reconstruction-free A.\n")

    torch.save({"state_dict": netA.state_dict(), "obs_mean": xmA, "obs_std": xsA},
               "il_actor_vec.pt")
    print("saved il_actor_vec.pt  (whole-vector policy)")


if __name__ == "__main__":
    main()
