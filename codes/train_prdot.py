"""
train_prdot.py — behavior-clone the optimizer's lambda from the distilled I/O.

Fits  [clock(14), pR_dot(n*3), lambda_{t-1}(n)] -> lambda[n]  on the optimizer data
collected by collect_prdot_data.py. Standardize, 80/20 split, best-val checkpoint,
report per-output R^2. Saves il_actor_prdot.pt.
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


def fit(X, Y):
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

    var_lambda = float(Y.var())                 # predict-the-mean baseline MSE
    tr_hist, va_hist = [], []
    best, best_ep, best_state = float("inf"), 0, None
    for ep in range(1, EPOCHS + 1):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); mse(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tr = mse(net(tro), trl).item()
            va = mse(net(vao), val).item()
        tr_hist.append(tr); va_hist.append(va)
        if va < best:
            best, best_ep = va, ep
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        print(f"ep {ep:3d}   train MSE {tr:.4f}   val MSE {va:.4f}")

    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pred = net(vao).cpu().numpy()
    yv = Y[vi]
    r2 = []
    for j in range(Y.shape[1]):
        ss_res = np.sum((yv[:, j] - pred[:, j]) ** 2)
        ss_tot = np.sum((yv[:, j] - yv[:, j].mean()) ** 2)
        r2.append(1.0 - ss_res / ss_tot)
    print(f"\nbest val MSE {best:.4f} at epoch {best_ep}   "
          f"R2 {[f'{r:.3f}' for r in r2]}   mean {np.mean(r2):.3f}")

    # Training curve.
    epochs = range(1, EPOCHS + 1)
    plt.figure()
    plt.plot(epochs, tr_hist, label="train MSE")
    plt.plot(epochs, va_hist, label="val MSE")
    plt.axhline(var_lambda, ls="--", c="gray", label="baseline = Var(lambda)")
    plt.axvline(best_ep, ls=":", c="green", label=f"best val (ep {best_ep})")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("pR_dot policy training curve (lambda regression)")
    plt.legend(); plt.grid(True)

    return net, xm.astype(np.float32), xs.astype(np.float32)


def main():
    d = np.load("prdot_dataset.npz")
    X, Y = d["X"], d["Y"]
    print(f"data: {len(X)} steps   X {X.shape}  Y {Y.shape}   Var(lambda)={Y.var():.4f}\n")

    net, xm, xs = fit(X, Y)

    torch.save({"state_dict": net.state_dict(), "obs_mean": xm, "obs_std": xs},
               "il_actor_prdot.pt")
    print("saved il_actor_prdot.pt  (pR_dot policy)")
    plt.show()


if __name__ == "__main__":
    main()
