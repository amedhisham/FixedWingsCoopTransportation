"""
demo_desync.py — run the F2 wrapper with ZERO residual actions and SEE what
decentralization desync does, with interactive main.py-style plots.

There is no learning here. We drive the 4-expert stitch with delta_f = 0 and turn
on the desync knobs, to check the disturbance actually degrades load tracking
(i.e. that the residual RL will have a real job to do).

Run:   python demo_desync.py
Edit the DESYNC dict below to change the disturbance. Set COMPARE=False to skip
the coherent baseline overlay (twice as fast).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv
from controller import get_reference_trajectory
from networks import Actor
from mappo import DESYNC as TRAIN_DESYNC, FIXED_DELAYS, FIXED_SEED   # exact training scenario

N = 4
EPSILON = 0.25
DESYNC_ON = True   # set False to see ONLY the coherent (no-noise, no-delay) baseline
COMPARE = False     # when desync is on, overlay a coherent run so degradation is obvious
POLICY_PATH = "residual_mappo_last.pt"   # trained residual to evaluate; None -> just base-only desync
                                         #   (_last = latest, saved on Ctrl+C; _fixed = fixed-scenario best)

# --- held-out generalization test: eval the trained policy on a scenario it NEVER trained on
#     (different noise seed + different delay assignment, same {1,2} range & noise levels). ---
GEN_SEED = 8888                     # != training FIXED_SEED(12345) and != training-eval EVAL_SEED(4242)
GEN_DELAYS = [2, 1, 1, 2]           # permuted delay assignment (trained on [1,2,2,1])

# Every disturbance knob is exposed here (all-zero == coherent baseline).
# Note: with ZERO residual actions, own_noise has no visible effect (no policy is
# reading the observation yet); load_noise / actuation_noise / delay / clock_offset do.
DESYNC = dict(
    ctrl_delay=[0, 2, 2, 1],                 # steps of load-estimate delay (x10 ms) per drone
    clock_offset=[0.0, 0.00, -0.00, 0.00],   # seconds added to each drone's clock
    pos_noise=0.03,                          # load position sensing noise, m   (shared expert + obs)
    rot_noise=0.01,                          # load orientation sensing noise
    vel_noise=0.10,                          # load linear-velocity sensing noise, m/s
    angvel_noise=0.05,                       # load angular-velocity sensing noise, rad/s
    noise_corr=0.995,                         # temporal smoothness of sensing noise (0=white, ->1=smooth)
    own_noise=0.0,                           # drone's own pos/vel sensing noise (obs only)
    actuation_noise=0.0,                     # noise on the commanded forces
)


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(obs_dim=ck["obs_dim"], act_dim=ck["act_dim"])
    actor.load_state_dict(ck["state_dict"]); actor.eval()
    return actor, ck["obs_mean"].astype(np.float32), ck["obs_std"].astype(np.float32)


def run_episode(policy=None, seed=None, **kwargs):
    """Roll out one episode. policy=None -> zero residual (base only); else the trained
    residual actor's deterministic MEAN action from each drone's local (noisy) obs.
    seed fixes the noise realization (pass FIXED_SEED to reproduce the training scenario)."""
    env = ResidualMARLEnv(n_carriers=N, epsilon=EPSILON, **kwargs)
    obs, _ = env.reset(seed=seed)
    agents = env.possible_agents

    t_hist, load_hist, ref_hist, swing_hist = [], [], [], []
    dpos = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]

    t = 0.0
    while True:
        s = env.state()                       # global 42-D state at current t
        t_hist.append(t)
        load_hist.append(s[0:3].copy())
        ref_p, ref_v, _, _ = get_reference_trajectory(t)
        ref_hist.append(ref_p.copy())
        swing_hist.append(np.linalg.norm(s[12:15] - ref_v))   # load velocity error = SWING rate
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
            act = {a: mean[i] for i, a in enumerate(agents)}
        obs, *_ = env.step(act)
        t += env.dt
        if not env.agents:                    # truncated -> episode over
            break
    env.close()
    return dict(
        t=np.array(t_hist), load=np.array(load_hist), ref=np.array(ref_hist),
        swing=np.array(swing_hist),
        dpos=[np.array(p) for p in dpos], dvel=[np.array(v) for v in dvel],
    )


if __name__ == "__main__":
    pol = load_policy(POLICY_PATH) if (POLICY_PATH and os.path.exists(POLICY_PATH)) else None
    if POLICY_PATH and pol is None:
        print(f"[warn] {POLICY_PATH} not found -> showing base-only desync")
    if pol is not None:                     # trained residual vs base-only on a HELD-OUT scenario
        cfg = dict(TRAIN_DESYNC, ctrl_delay=GEN_DELAYS)   # different delays + seed than training
        des = run_episode(policy=pol, seed=GEN_SEED, **cfg)
        coh = run_episode(seed=GEN_SEED, **cfg)
        print(f"[held-out] seed={GEN_SEED} delays={GEN_DELAYS}  (trained on seed={FIXED_SEED} delays={FIXED_DELAYS})")
        lbl_des, lbl_coh = "trained residual", "base (no residual)"
    elif DESYNC_ON:
        des = run_episode(**DESYNC)
        coh = run_episode() if COMPARE else None
        lbl_des, lbl_coh = "desynced", "coherent"
    else:
        des = run_episode(); coh = None
        lbl_des, lbl_coh = "load", None

    e_des = np.linalg.norm(des["load"] - des["ref"], axis=1)
    print(f"{lbl_des:20s}: mean track {e_des.mean():.4f} m   max {e_des.max():.4f} m")
    if coh is not None:
        e_coh = np.linalg.norm(coh["load"] - coh["ref"], axis=1)
        print(f"{lbl_coh:20s}: mean track {e_coh.mean():.4f} m   max {e_coh.max():.4f} m")

    # 1. Drone velocity norms (trained solid; base faded).
    plt.figure()
    for i in range(N):
        plt.plot(des["t"], des["dvel"][i], lw=1.4, label=f"Drone {i+1}")
        if coh is not None:
            plt.plot(coh["t"], coh["dvel"][i], color=f"C{i}", lw=1.0, alpha=0.25)
    plt.axhline(EPSILON, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms — {lbl_des} (faded = {lbl_coh})"); plt.legend(); plt.grid(True)

    # 2. Load position vs reference, per axis.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(des["t"], des["ref"][:, k], "k--", lw=2, label="reference")
        ax.plot(des["t"], des["load"][:, k], "b", label=lbl_des)
        if coh is not None:
            ax.plot(coh["t"], coh["load"][:, k], "r", alpha=0.7, label=lbl_coh)
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)"); fig.suptitle("Load position tracking")

    # 3. Load tracking-error norm over time.
    plt.figure()
    plt.plot(des["t"], e_des, "b", label=lbl_des)
    if coh is not None:
        plt.plot(coh["t"], np.linalg.norm(coh["load"] - coh["ref"], axis=1), "r", alpha=0.7, label=lbl_coh)
    plt.xlabel("Time (s)"); plt.ylabel("||load - reference|| (m)")
    plt.title("Load tracking error"); plt.legend(); plt.grid(True)

    # 3b. Load SWING = velocity-error norm over time (what the swing reward targets).
    plt.figure()
    plt.plot(des["t"], des["swing"], "b", label=lbl_des)
    if coh is not None:
        plt.plot(coh["t"], coh["swing"], "r", alpha=0.7, label=lbl_coh)
    print(f"{lbl_des:20s}: mean swing {des['swing'].mean():.4f}   max {des['swing'].max():.4f} m/s")
    if coh is not None:
        print(f"{lbl_coh:20s}: mean swing {coh['swing'].mean():.4f}   max {coh['swing'].max():.4f} m/s")
    plt.xlabel("Time (s)"); plt.ylabel("||load vel - ref vel|| (m/s)")
    plt.title("Load swing (velocity error)"); plt.legend(); plt.grid(True)

    # 4. Drone XY trajectories (trained solid) overlaid on the EXPERT loiter loop (faded).
    try:
        expert = np.load("expert_ref.npz")["dpos"]   # (N,T,3) the manifold the reward tracks
    except FileNotFoundError:
        expert = None
    plt.figure(figsize=(8, 6))
    for i in range(N):
        if expert is not None:
            plt.plot(expert[i][:, 0], expert[i][:, 1], color=f"C{i}", lw=3.0, alpha=0.25)
        plt.plot(des["dpos"][i][:, 0], des["dpos"][i][:, 1], color=f"C{i}", lw=1.2, label=f"Drone {i+1}")
    plt.plot(des["load"][:, 0], des["load"][:, 1], "k--", lw=2, label="Load")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title(f"Drone XY — {lbl_des} (faded = expert loop)")
    plt.legend(); plt.grid(True); plt.axis("equal")

    plt.show()
