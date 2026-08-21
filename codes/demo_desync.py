"""
demo_desync.py — F2 comparison plots on SHOWCASE trajectories, ONE at a time.

The load lives on range(G) (served by f_g = G+ w_d, from the PID) so it tracks no matter how badly
the drones' NULLSPACE config is doing -> load-tracking hides the failure. The story is the DRONE XY
/ velocity (nullspace), so the load plots show the RL run vs the trajectory reference only.

Two modes (like deploy_compare). The protagonist is always the trained RL residual UNDER NOISE;
only the comparison baseline changes. Each run gets its OWN full standalone figures (drone XY + load,
velocity) — no faded overlay — so you compare the RL plots against the baseline plots side by side:
  PURE  : RL-under-noise  vs  the IDEAL (central expert, NO noise)  -> "we match the ideal".
  NOISY : RL-under-noise  vs  the BASE controller UNDER NOISE (no residual) -> "the raw controller
                                                                              is COOKED; RL recovers".

Pick the trajectory with SHOWCASE_KIND ('short'/'long') + SHOWCASE_IDX (deploy_prdot-style):
  idx 0 = the straight line; idx 1.. = the quintic-pose demos. 'long' (35 s) gives the base more
  time to diverge -> the spiral is more obvious.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv
from controller import get_reference_trajectory
from collect_il_data import DT
from networks import Actor
from expert_reference import expert_path
from trajectories import showcase_set
from mappo import DESYNC as TRAIN_DESYNC          # the training noise levels

N = 4
EPSILON = 0.25
MODE = "NOISY"             # "PURE" (vs ideal, no noise) | "NOISY" (vs base controller under noise)
SHOWCASE_KIND = "long"    # "short" (25 s) | "long" (35 s)
SHOWCASE_IDX = 0          # trajectory in showcase_set(KIND): 0 = line, 1.. = quintics
POLICY_PATH = "residual_mappo_r4base.pt"   # trained residual to evaluate
ZERO_DW = False           # zero the delta_wrench head at apply -> delta_lambda-only ablation

# Held-out disturbance scenario (never trained on): different noise seed + delay assignment.
GEN_SEED = 8888
GEN_DELAYS = [2, 2, 2, 2]   


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(obs_dim=ck["obs_dim"], act_dim=ck["act_dim"])
    actor.load_state_dict(ck["state_dict"]); actor.eval()
    return actor, ck["obs_mean"].astype(np.float32), ck["obs_std"].astype(np.float32)


def run_episode(policy=None, seed=None, traj=None, end_time=25.0, zero_dw=False, **kwargs):
    """One ResidualMARLEnv episode on `traj` (env + reference threaded). policy=None -> base
    (zero residual); else the trained residual's deterministic mean from each drone's local obs.
    Returns a plot dict {t, load, ref, dpos, dvel(norms)}."""
    env = ResidualMARLEnv(n_carriers=N, epsilon=EPSILON, traj=traj, end_time=end_time, **kwargs)
    obs, _ = env.reset(seed=seed)
    agents = env.possible_agents

    t_hist, load_hist, ref_hist = [], [], []
    dpos = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    t = 0.0
    while True:
        s = env.state()
        t_hist.append(t)
        load_hist.append(s[0:3].copy())
        ref_hist.append(get_reference_trajectory(t, traj)[0].copy())
        for i in range(N):
            dpos[i].append(s[18 + 3 * i: 18 + 3 * i + 3].copy())
            dvel[i].append(np.linalg.norm(s[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        if policy is None:
            act = {a: np.zeros(env._act_space.shape[0], dtype=np.float32) for a in agents}
        else:
            actor, om, os_ = policy
            oa = np.stack([obs[a] for a in agents]).astype(np.float32)
            with torch.no_grad():
                mean = actor(torch.tensor(((oa - om) / os_).astype(np.float32))).numpy()
            if zero_dw:
                mean[:, N:N + 6] = 0.0            # delta_lambda-only: drop the load-trim head
            act = {a: mean[i] for i, a in enumerate(agents)}
        obs, *_ = env.step(act)
        t += env.dt
        if not env.agents:                       # truncated (episode end OR blowup guard)
            break
    env.close()
    return dict(t=np.array(t_hist), load=np.array(load_hist), ref=np.array(ref_hist),
                dpos=[np.array(p) for p in dpos], dvel=[np.array(v) for v in dvel])


def expert_run(traj, t_end):
    """The IDEAL (central expert, NO noise) as a plot dict compatible with run_episode's output."""
    dpos, dvel, load = expert_path(traj, t_end)
    t = np.arange(load.shape[0]) * DT
    ref = np.array([get_reference_trajectory(tt, traj)[0] for tt in t])
    return dict(t=t, load=load, ref=ref,
                dpos=[dpos[i] for i in range(N)],
                dvel=[np.linalg.norm(dvel[i], axis=1) for i in range(N)])


if __name__ == "__main__":
    label, traj, t_end = showcase_set(SHOWCASE_KIND)[SHOWCASE_IDX]
    if not (POLICY_PATH and os.path.exists(POLICY_PATH)):
        raise SystemExit(f"POLICY_PATH {POLICY_PATH!r} not found — the demo needs the trained residual.")
    pol = load_policy(POLICY_PATH)

    noisy_cfg = dict(TRAIN_DESYNC, ctrl_delay=GEN_DELAYS)
    # protagonist (both modes): the trained residual UNDER NOISE.
    rl = run_episode(policy=pol, seed=GEN_SEED, traj=traj, end_time=t_end, zero_dw=ZERO_DW, **noisy_cfg)
    lbl_rl = "RL residual (dlam-only, noise)" if ZERO_DW else "RL residual (noise)"

    if MODE == "PURE":
        cmp_ = expert_run(traj, t_end)                                   # ideal, NO noise
        lbl_cmp = "ideal (central expert, no noise)"
    elif MODE == "NOISY":
        cmp_ = run_episode(policy=None, seed=GEN_SEED, traj=traj,        # base controller, WITH noise
                           end_time=t_end, **noisy_cfg)
        lbl_cmp = "base controller (noise)"
    else:
        raise ValueError(f"unknown MODE {MODE!r} (use 'PURE' or 'NOISY')")

    e_rl = np.linalg.norm(rl["load"] - rl["ref"], axis=1)
    print(f"[{MODE}] showcase {SHOWCASE_KIND}#{SHOWCASE_IDX} = {label}  t_end {t_end}s  "
          f"(held-out seed {GEN_SEED}, delays {GEN_DELAYS})")
    print(f"  {lbl_rl:32s}: load-track mean {e_rl.mean():.4f} m  max {e_rl.max():.4f} m")
    print("  (no base/ideal LOAD line: load rides range(G), fine either way — the failure is in the drones)")

    e_cmp = np.linalg.norm(cmp_["load"] - cmp_["ref"], axis=1)
    print(f"  {lbl_cmp:32s}: load-track mean {e_cmp.mean():.4f} m  max {e_cmp.max():.4f} m")

    # 1. Drone XY — RL solid, comparison FADED. NOISY -> RL loops vs base SPIRAL; PURE -> vs ideal loops.
    plt.figure(figsize=(8, 6))
    for i in range(N):
        plt.plot(cmp_["dpos"][i][:, 0], cmp_["dpos"][i][:, 1], color=f"C{i}", lw=2.6, alpha=0.28)
        plt.plot(rl["dpos"][i][:, 0], rl["dpos"][i][:, 1], color=f"C{i}", lw=1.3, label=f"Drone {i+1}")
    plt.plot(rl["load"][:, 0], rl["load"][:, 1], "k--", lw=2, label="Load")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title(f"Drone XY — {lbl_rl} (solid) vs {lbl_cmp} (faded)\n{SHOWCASE_KIND} #{SHOWCASE_IDX}: {label}")
    plt.legend(); plt.grid(True); plt.axis("equal")

    # 2. Drone velocity norms — RL solid, comparison FADED. NOISY base -> spikes/tension collapse.
    plt.figure()
    for i in range(N):
        plt.plot(rl["t"], rl["dvel"][i], color=f"C{i}", lw=1.4, label=f"Drone {i+1}")
        plt.plot(cmp_["t"], cmp_["dvel"][i], color=f"C{i}", lw=1.0, alpha=0.3)
    plt.axhline(EPSILON, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms — {lbl_rl} (solid) vs {lbl_cmp} (faded)")
    plt.legend(); plt.grid(True)

    # 3. Load position tracking per axis — RL vs the TRAJECTORY reference (the "RL keeps load on track" story).
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, axlbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(rl["t"], rl["ref"][:, k], "k--", lw=2, label="reference")
        ax.plot(rl["t"], rl["load"][:, k], "b", label=lbl_rl)
        ax.set_ylabel(f"{axlbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)"); fig.suptitle(f"Load tracking — {lbl_rl} (RL keeps the load on track)")

    # 4. Load tracking-error norm over time — RL only.
    plt.figure()
    plt.plot(rl["t"], e_rl, "b", label=lbl_rl)
    plt.xlabel("Time (s)"); plt.ylabel("||load - reference|| (m)")
    plt.title("Load tracking error (RL residual)"); plt.legend(); plt.grid(True)

    # 5. (curiosity) The LOAD's OWN XY path — separate so you can see what the load itself does under
    #    the comparison baseline (noised controller in NOISY) vs RL vs the reference, uncluttered by drones.
    plt.figure(figsize=(8, 6))
    plt.plot(rl["ref"][:, 0], rl["ref"][:, 1], "k--", lw=2, label="reference")
    plt.plot(cmp_["load"][:, 0], cmp_["load"][:, 1], "r", lw=1.4, label=f"load — {lbl_cmp}")
    plt.plot(rl["load"][:, 0], rl["load"][:, 1], "b", lw=1.4, label=f"load — {lbl_rl}")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("Load XY trajectory"); plt.legend(); plt.grid(True); plt.axis("equal")

    plt.show()
