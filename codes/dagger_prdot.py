"""
dagger_prdot.py — DAgger for the pR_dot whole-vector lambda policy.

The BC policy (il_actor_prdot.pt) fit the optimizer perfectly (R^2=1.0) but buzzes
in closed loop: it reconstructs pR_dot from its OWN lambda_{t-1}, which drifts off
the manifold the OPTIMIZER's lambda produced during collection. Classic covariate
shift — the self-fed input distribution was never trained on.

DAgger fixes it: roll out the CURRENT policy, but build its input (pR_dot + lambda_{t-1})
from the ACTUALLY-APPLIED lambda history (the state it really visits), label every
visited state with the EXPERT (optimizer) lambda, aggregate, retrain, repeat. As
beta -> 0 the applied lambda -> the policy's own, so the training distribution
converges to the deployment distribution.

Two enablers, same as F1:
  - The clock keeps the off-manifold labels a consistent function of the input.
  - G.N = 0 makes beta-MIXING safe: applied lambda = beta*expert + (1-beta)*policy
    only touches the nullspace, so any blend leaves load tracking intact.

Whole-vector: ONE 30-D input row -> one lambda[N] label per step (not 4 per-drone rows).
Saves il_actor_prdot_dagger.pt, prints per-iter retrain MSE + buzz, then runs
deploy_prdot on the final policy.
"""

import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from optimizer import cable_force_calculation
from controller import error_calculation, get_reference_trajectory
from networks import Actor
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from collect_prdot_data import Reconstructor, build_input, LAM0, SUFFIX
from deploy_prdot import main as deploy_prdot_main

BYPASS_OPT = False   # adaptive optimizer (matches prdot_dataset.npz)

# beta schedule: 1 = pure expert (stay on the optimizer manifold), 0 = pure policy
# (deployment distribution). Decay collects progressively more off-manifold states.
BETAS = [0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]

EPOCHS = 250
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.2
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-iteration plots auto-close after this many seconds so the run proceeds
# unattended. Set 0 to block on every iteration. Final buzz curve + deploy block.
PAUSE_SEC = 20.0


def rollout(policy, om, os_, beta):
    """Drive the plant with lambda = beta*expert + (1-beta)*policy for one episode.

    The policy input (pR_dot + lambda_{t-1}) is reconstructed from the APPLIED
    (mixed) lambda history — the state distribution the policy actually visits.
    Returns visited-state inputs (M,30) labelled with EXPERT lambda (M,N) + diag.
    """
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)   # fresh expert state

    prev_f = np.array([0.0, 0.0, FZ] * N)
    prev_lam = LAM0.copy()          # APPLIED lambda_{t-1}
    recon = Reconstructor(Bb, L0, DT)

    X_rows, Y_rows = [], []
    lam_pol_hist = [[] for _ in range(N)]
    lam_exp_hist = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    load_hist, ref_hist = [], []

    t = 0.0
    while t < T_END - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        # w_d ONCE (advances integrators once); reused for input, expert, and drive.
        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)

        # Policy input at the VISITED state: pR_dot (mode set by ANALYTIC) from lambda_{t-1}.
        vR = recon(R, vel, angvel, w_d, prev_lam)
        row = build_input(t, vR, prev_lam)
        Xn = ((row[None, :] - om) / os_).astype(np.float32)
        with torch.no_grad():
            lam_pol = policy(torch.tensor(Xn)).numpy().flatten()

        # Expert label at the SAME visited state = the DAgger target.
        lam_exp, _ = agent.optimize(t, R, vel, angvel, w_d, bypass=BYPASS_OPT)

        # Rollout driven by the MIX; label is the pure EXPERT.
        lam_mixed = beta * lam_exp + (1.0 - beta) * lam_pol
        f_full, _ = cable_force_calculation(R, Bb, w_d, lam_mixed, N)

        X_rows.append(row)
        Y_rows.append(lam_exp)
        for i in range(N):
            lam_pol_hist[i].append(lam_pol[i])
            lam_exp_hist[i].append(lam_exp[i])
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        load_hist.append(pos.copy())
        ref_hist.append(get_reference_trajectory(t)[0].copy())

        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))

        # Roll histories with the APPLIED (mixed) lambda.
        recon.roll(lam_mixed)
        prev_lam = lam_mixed.copy()
        t += DT
    env.close()

    lam_pol_hist = [np.array(l) for l in lam_pol_hist]
    lam_exp_hist = [np.array(l) for l in lam_exp_hist]
    dvel = [np.array(v) for v in dvel]
    load = np.array(load_hist); ref = np.array(ref_hist)
    buzz = float(np.mean([np.mean(np.abs(np.diff(l))) for l in lam_pol_hist]))
    diag = {
        "buzz": buzz,
        "vmin": float(min(v.min() for v in dvel)),
        "vmean": float(np.mean([v.mean() for v in dvel])),
        "track_mean": float(np.linalg.norm(load - ref, axis=1).mean()),
        "track_max": float(np.linalg.norm(load - ref, axis=1).max()),
        "lam_pol": lam_pol_hist,
        "lam_exp": lam_exp_hist,
        "dvel": dvel,
        "t": np.arange(len(load)) * DT,
    }
    return (np.asarray(X_rows, dtype=np.float32),
            np.asarray(Y_rows, dtype=np.float32),
            diag)


def train(X, Y):
    """Retrain from scratch on the aggregated set. Fresh normalization, 80/20 split,
    best-val checkpoint. Returns cpu state_dict + stats + best-val MSE + Var(lambda)."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    xm = X.mean(0, keepdims=True)
    xs = X.std(0, keepdims=True) + 1e-6
    Xn = ((X - xm) / xs).astype(np.float32)

    M = len(X)
    idx = rng.permutation(M)
    n_val = int(VAL_FRAC * M)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tt = lambda a: torch.tensor(a, device=DEVICE)
    tr_x, tr_y = tt(Xn[tr_idx]), tt(Y[tr_idx])
    va_x, va_y = tt(Xn[val_idx]), tt(Y[val_idx])
    loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=BATCH, shuffle=True)

    net = Actor(obs_dim=X.shape[1], act_dim=Y.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    best_va, best_state = float("inf"), None
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); mse(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = mse(net(va_x), va_y).item()
        if va < best_va:
            best_va = va
            best_state = copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()})
    return best_state, xm.astype(np.float32), xs.astype(np.float32), best_va, float(Y.var())


def show_diag(diag, label):
    """Overlay expert-vs-policy lambda + velocity norms for one rollout."""
    fig1, ax1 = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(ax1):
        ax.plot(diag["t"], diag["lam_exp"][i], "k", lw=1.0, label="expert (would-do)")
        ax.plot(diag["t"], diag["lam_pol"][i], "m", lw=1.0, alpha=0.8, label="policy")
        ax.set_ylabel(f"$\\lambda_{i+1}$"); ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    ax1[-1].set_xlabel("Time (s)"); fig1.suptitle(f"Expert vs policy lambda — {label}")

    plt.figure()
    for i in range(N):
        plt.plot(diag["t"], diag["dvel"][i], label=f"Drone {i+1}")
    plt.axhline(EPS, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms — {label}"); plt.legend(); plt.grid(True)

    if PAUSE_SEC > 0:
        plt.show(block=False)
        plt.pause(PAUSE_SEC)
        plt.close("all")
    else:
        plt.show()


def main():
    # Warm-start from the BC prdot policy and its dataset.
    ckpt = torch.load(f"il_actor_prdot{SUFFIX}.pt", map_location="cpu", weights_only=False)
    policy = Actor(obs_dim=ckpt["obs_mean"].shape[1], act_dim=N)
    policy.load_state_dict(ckpt["state_dict"]); policy.eval()
    om, os_ = ckpt["obs_mean"].astype(np.float32), ckpt["obs_std"].astype(np.float32)

    data = np.load(f"prdot_dataset{SUFFIX}.npz")
    D_X = data["X"].astype(np.float32)
    D_Y = data["Y"].astype(np.float32)

    buzz_curve = []
    for k, beta in enumerate(BETAS, 1):
        new_X, new_Y, diag = rollout(policy, om, os_, beta)
        show_diag(diag, f"iter {k} rollout  (beta {beta:.2f})")

        D_X = np.concatenate([D_X, new_X], axis=0)
        D_Y = np.concatenate([D_Y, new_Y], axis=0)
        state, om, os_, best_va, var_lam = train(D_X, D_Y)
        policy = Actor(obs_dim=D_X.shape[1], act_dim=N)
        policy.load_state_dict(state); policy.eval()

        buzz_curve.append(diag["buzz"])
        print(f"iter {k}  beta {beta:.2f}  |  rollout buzz {diag['buzz']:.4f}  "
              f"vmin {diag['vmin']:.3f}  track {diag['track_mean']:.4f}  |  "
              f"retrain best-val MSE {best_va:.4f} (Var lam {var_lam:.3f})  |  "
              f"dataset {len(D_X)}")

    torch.save({"state_dict": {k: v for k, v in policy.state_dict().items()},
                "obs_mean": om, "obs_std": os_}, f"il_actor_prdot_dagger{SUFFIX}.pt")
    print(f"\nsaved il_actor_prdot_dagger{SUFFIX}.pt")

    plt.figure()
    plt.plot(range(1, len(buzz_curve) + 1), buzz_curve, "o-")
    plt.xlabel("DAgger iteration"); plt.ylabel("policy action buzz  (mean |d lambda/step|)")
    plt.title("Closed-loop lambda buzz across DAgger"); plt.grid(True)
    plt.show()

    print(f"\n--- deploy_prdot on il_actor_prdot_dagger{SUFFIX}.pt ---")
    deploy_prdot_main(f"il_actor_prdot_dagger{SUFFIX}.pt")


if __name__ == "__main__":
    main()
