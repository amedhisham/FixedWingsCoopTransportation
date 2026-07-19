"""
dagger.py — DAgger for the Formulation-1 lambda policy.

The BC policy (il_actor.pt) only ever saw ON-orbit states, so in closed loop it
drifts off that distribution and buzzes (covariate shift + the nullspace loop has
no restoring feedback). DAgger fixes the covariate-shift half: roll out the
CURRENT policy, label every state it actually visits with the expert's lambda (a
free query here — the expert is an algorithm), aggregate, retrain, repeat.

Two things make this work in THIS system:
  - The clock in the obs makes the off-orbit labels a CONSISTENT function of the
    observation (no "same position, different time -> different lambda" collision),
    so the aggregated off-orbit data is learnable. Without the clock DAgger would
    floor at Var(lambda). This is why we added the clock first.
  - G.N = 0 makes beta-MIXING safe: applied lambda = beta*expert + (1-beta)*policy
    only touches the nullspace, so any blend leaves load tracking intact. We start
    beta high (stay near the good orbit) and decay to 0 (the deployment
    distribution), collecting the "slightly off, needs to recover" states.

Labels use the EXPERT lambda at each visited state; the rollout is driven by the
mixed lambda. Saves il_actor_dagger.pt. Prints per-iteration retrain MSE + a buzz
metric (mean |d lambda/step| of the policy's own action) so we watch the closed
-loop jitter shrink, then shows the final beta=0 lambda panel + load tracking.
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
from collect_il_data import build_obs_rows, read_params
from deploy_f1 import main as deploy_f1_main

N = 4
DT, T_END = 0.01, 25.0
EPS = 0.25
PHASES = np.array([0.0, np.pi / 2, 0.0, np.pi / 2])
LLC_ALPHA = DT / (0.2 + DT)
FZ = 0.7 * 9.81 / 4

# Must match how il_dataset.npz was collected (adaptive optimizer).
BYPASS_OPT = False

# DAgger beta schedule: 1 = pure expert (stay on orbit), 0 = pure policy
# (deployment distribution). Decay collects progressively more off-orbit states.
BETAS = [0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]

# Per-iteration retrain (from scratch, fresh normalization over the aggregated set).
EPOCHS = 250
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.2
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-iteration plots auto-close after this many seconds so the run proceeds
# unattended. Set 0 to block on every iteration (inspect and close manually).
# The final buzz curve + deploy_f1 plots always block, so results wait for you.
PAUSE_SEC = 20.0


def rollout(policy, obs_mean, obs_std, beta):
    """Drive the plant with lambda = beta*expert + (1-beta)*policy for one episode.

    Returns:
      new_obs (M,38), new_lam (M,1)   -- visited states labelled with EXPERT lambda,
      diag dict                       -- policy-action buzz, velocity norms, tracking.
    """
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)   # fresh integrators/optimizer history

    prev_f = np.array([0.0, 0.0, FZ] * N)
    new_obs, new_lam = [], []
    lam_pol_hist = [[] for _ in range(N)]      # the POLICY's raw action
    lam_exp_hist = [[] for _ in range(N)]      # the EXPERT's lambda at the SAME (policy) states
    dvel = [[] for _ in range(N)]
    load_hist, ref_hist = [], []

    om = obs_mean.astype(np.float32)
    os_ = obs_std.astype(np.float32)

    t = 0.0
    while t < T_END - 1e-9:
        rows = build_obs_rows(obs42, t)
        X = (np.stack(rows) - om) / os_
        with torch.no_grad():
            lam_pol = policy(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        # Expert pieces — wrench controller stepped exactly ONCE (integrators), then
        # the optimizer's lambda at this actually-visited state = the DAgger label.
        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)
        lam_exp, _ = agent.optimize(t, R, vel, angvel, w_d, bypass=BYPASS_OPT)

        # Rollout is driven by the MIX; labels are the pure EXPERT.
        lam_mixed = beta * lam_exp + (1.0 - beta) * lam_pol
        f_full, _ = cable_force_calculation(R, Bb, w_d, lam_mixed, N)

        for i in range(N):
            new_obs.append(rows[i])
            new_lam.append(lam_exp[i])
            lam_pol_hist[i].append(lam_pol[i])
            lam_exp_hist[i].append(lam_exp[i])
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        load_hist.append(pos.copy())
        ref_hist.append(get_reference_trajectory(t)[0].copy())

        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    lam_pol_hist = [np.array(l) for l in lam_pol_hist]
    lam_exp_hist = [np.array(l) for l in lam_exp_hist]
    dvel = [np.array(v) for v in dvel]
    load = np.array(load_hist); ref = np.array(ref_hist)
    # buzz = mean over drones of mean absolute step-to-step change in the policy action.
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
    return (np.asarray(new_obs, dtype=np.float32),
            np.asarray(new_lam, dtype=np.float32).reshape(-1, 1),
            diag)


def train(obs, lam):
    """Retrain from scratch on the aggregated set. Fresh normalization, 80/20
    split, best-val checkpoint. Returns cpu state_dict + normalization stats."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    obs_mean = obs.mean(0, keepdims=True)
    obs_std = obs.std(0, keepdims=True) + 1e-6
    obs_n = ((obs - obs_mean) / obs_std).astype(np.float32)

    M = len(obs)
    idx = rng.permutation(M)
    n_val = int(VAL_FRAC * M)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tt = lambda a: torch.tensor(a, device=DEVICE)
    tr_obs, tr_lam = tt(obs_n[tr_idx]), tt(lam[tr_idx])
    va_obs, va_lam = tt(obs_n[val_idx]), tt(lam[val_idx])
    loader = DataLoader(TensorDataset(tr_obs, tr_lam), batch_size=BATCH, shuffle=True)

    net = Actor(obs_dim=obs.shape[1], act_dim=1).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    best_va, best_state = float("inf"), None
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); mse(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = mse(net(va_obs), va_lam).item()
        if va < best_va:
            best_va = va
            best_state = copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()})
    return best_state, obs_mean.astype(np.float32), obs_std.astype(np.float32), best_va, float(lam.var())


def show_diag(diag, label):
    """Overlay expert-vs-policy lambda + velocity norms for one rollout, then block.
    Called every iteration so we watch convergence live."""
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
    # Start from the BC warm-start and its data.
    ckpt = torch.load("il_actor.pt", map_location="cpu", weights_only=False)
    policy = Actor(obs_dim=ckpt["obs_mean"].shape[1], act_dim=1)
    policy.load_state_dict(ckpt["state_dict"]); policy.eval()
    obs_mean, obs_std = ckpt["obs_mean"].astype(np.float32), ckpt["obs_std"].astype(np.float32)

    data = np.load("il_dataset.npz")
    D_obs = data["obs"].astype(np.float32)
    D_lam = data["lam"].astype(np.float32)

    buzz_curve = []
    for k, beta in enumerate(BETAS, 1):
        # ONE rollout — it is the DAgger data-collection rollout (the optimizer runs to
        # produce the expert LABELS we train on). We simply plot the expert/policy
        # lambdas it already computed. No extra rollout, no extra CasADi for the plot.
        new_obs, new_lam, diag = rollout(policy, obs_mean, obs_std, beta)
        show_diag(diag, f"iter {k} rollout  (beta {beta:.2f})")

        D_obs = np.concatenate([D_obs, new_obs], axis=0)
        D_lam = np.concatenate([D_lam, new_lam], axis=0)
        state, obs_mean, obs_std, best_va, var_lam = train(D_obs, D_lam)
        policy = Actor(obs_dim=D_obs.shape[1], act_dim=1)
        policy.load_state_dict(state); policy.eval()

        buzz_curve.append(diag["buzz"])
        print(f"iter {k}  beta {beta:.2f}  |  rollout buzz {diag['buzz']:.4f}  "
              f"vmin {diag['vmin']:.3f}  track {diag['track_mean']:.4f}  |  "
              f"retrain best-val MSE {best_va:.4f} (Var lam {var_lam:.3f})  |  "
              f"dataset {len(D_obs)}")

    torch.save({"state_dict": {k: v for k, v in policy.state_dict().items()},
                "obs_mean": obs_mean, "obs_std": obs_std}, "il_actor_dagger.pt")
    print("\nsaved il_actor_dagger.pt")

    # Buzz vs iteration (each point is that iteration's rollout, at its schedule beta).
    plt.figure()
    plt.plot(range(1, len(buzz_curve) + 1), buzz_curve, "o-")
    plt.xlabel("DAgger iteration"); plt.ylabel("policy action buzz  (mean |d lambda/step|)")
    plt.title("Closed-loop lambda buzz across DAgger"); plt.grid(True)
    plt.show()

    # Final deployment view of the DAgger policy (deploy_f1 is CasADi-free: policy
    # replaces the optimizer). Load tracking, velocity norms, XY paths, lambda action.
    print("\n--- deploy_f1 on il_actor_dagger.pt ---")
    deploy_f1_main("il_actor_dagger.pt")


if __name__ == "__main__":
    main()
