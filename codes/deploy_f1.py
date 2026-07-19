"""
deploy_f1.py — closed-loop test of the imitation-learned Formulation-1 policy.

Replaces the OPTIMIZER with the trained lambda-network, keeping the wrench
controller and the nullspace distribution:

    w_d   = wrench_controller(load state)          # kept (analytic)
    lambda_i = policy(local obs_i)                  # LEARNED, replaces optimizer
    f      = G^dagger w_d + N lambda                # kept (G.N = 0 guarantees w_d)
    -> LLC -> plant

Open-loop MSE being low does NOT guarantee this flies (errors compound), so we
run the real plant and look at:
  - load tracking (should hold by construction, for ANY lambda),
  - drone velocity norms (does the learned loiter keep them moving?),
  - drone trajectories (sensible ellipses, or jitter?).
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from optimizer import cable_force_calculation
from controller import error_calculation, get_reference_trajectory
from networks import Actor
from collect_il_data import build_obs_rows, read_params

N = 4
DT, T_END = 0.01, 25.0
EPS = 0.25
PHASES = np.array([0.0, np.pi / 2, 0.0, np.pi / 2])
LLC_ALPHA = DT / (0.2 + DT)
FZ = 0.7 * 9.81 / 4
DEFAULT_POLICY = "il_actor_dagger.pt"   # single source of truth — change here only


def load_policy(path=DEFAULT_POLICY):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own file (has numpy stats)
    net = Actor(obs_dim=ckpt["obs_mean"].shape[1], act_dim=1)        # obs_dim follows the checkpoint
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["obs_mean"], ckpt["obs_std"]


def main(policy_path=DEFAULT_POLICY):
    net, obs_mean, obs_std = load_policy(policy_path)

    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)   # used only for wrench_control

    prev_f = np.array([0.0, 0.0, FZ] * N)
    t_hist, load_hist, ref_hist = [], [], []
    dpos = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    lam_hist = [[] for _ in range(N)]

    t = 0.0
    loop_t0 = time.perf_counter()
    while t < T_END - 1e-9:
        # --- policy lambda from the 4 local observations (deterministic mean) ---
        rows = build_obs_rows(obs42, t)
        X = (np.stack(rows) - obs_mean) / obs_std           # (4, obs_dim), same normalization as training
        with torch.no_grad():
            lam = net(torch.tensor(X, dtype=torch.float32)).numpy().flatten()   # (4,)

        # --- kept analytic pieces: wrench controller + nullspace distribution ---
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]
        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)
        f_full, _ = cable_force_calculation(R, Bb, w_d, lam, N)   # f = G^+ w_d + N lambda

        # --- LLC -> plant ---
        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()

        # record
        t_hist.append(t)
        load_hist.append(pos.copy())
        ref_hist.append(get_reference_trajectory(t)[0].copy())
        for i in range(N):
            dpos[i].append(obs42[18 + 3 * i: 18 + 3 * i + 3].copy())
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
            lam_hist[i].append(lam[i])

        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    loop_time = time.perf_counter() - loop_t0
    env.close()

    t_hist = np.array(t_hist); load = np.array(load_hist); ref = np.array(ref_hist)
    dpos = [np.array(p) for p in dpos]
    dvel = [np.array(v) for v in dvel]
    lam_hist = [np.array(l) for l in lam_hist]

    # 1. Load position tracking.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(t_hist, ref[:, k], "k--", lw=2, label="reference")
        ax.plot(t_hist, load[:, k], "b", label="policy")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)"); fig.suptitle("Load tracking — IL policy driving lambda")

    # 2. Drone velocity norms.
    plt.figure()
    for i in range(N):
        plt.plot(t_hist, dvel[i], label=f"Drone {i+1}")
    plt.axhline(EPS, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title("Drone velocity norms — IL policy"); plt.legend(); plt.grid(True)

    # 3. Drone XY trajectories.
    plt.figure(figsize=(8, 6))
    for i in range(N):
        plt.plot(dpos[i][:, 0], dpos[i][:, 1], label=f"Drone {i+1}")
    plt.plot(load[:, 0], load[:, 1], "k--", lw=2, label="Load")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("Drone XY trajectories — IL policy"); plt.legend(); plt.grid(True); plt.axis("equal")

    # 4. Policy action (lambda) per drone.
    fig4, ax4 = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(ax4):
        ax.plot(t_hist, lam_hist[i], "m")
        ax.set_ylabel(f"$\\lambda_{i+1}$ (action)"); ax.grid(True)
    ax4[-1].set_xlabel("Time (s)"); fig4.suptitle("Policy action (lambda) — IL policy (closed loop)")

    # Numeric summary.
    n_steps = len(t_hist)
    print(f"sim loop: {loop_time:.3f} s for {n_steps} steps  "
          f"({1000 * loop_time / n_steps:.3f} ms/step, {T_END / loop_time:.1f}x real-time)")
    err = np.linalg.norm(load - ref, axis=1)
    print(f"mean load tracking error = {err.mean():.4f} m   max = {err.max():.4f} m")
    for i in range(N):
        print(f"drone {i+1}: velocity norm min {dvel[i].min():.3f}  mean {np.mean(dvel[i]):.3f}")
    plt.show()


if __name__ == "__main__":
    main()
