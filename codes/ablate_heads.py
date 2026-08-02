"""
ablate_heads.py — does delta_wrench (range head) actually help, or is all the
benefit from delta_lambda (nullspace head)?

Loads a trained two-head residual policy and runs the SAME deterministic held-out
eval as mappo.eval_policy (EVAL_SEED / EVAL_DELAYS -> numbers comparable to the
DET_R training log), under four action modes:
  full     : policy output unchanged            [dlam, dw]
  no_dw    : zero the delta_wrench dims          [dlam, 0]
  no_dlam  : zero the delta_lambda dims          [0,    dw]
  base     : zero everything (pure f_base)       [0,    0]
Also reports the mean L2 magnitude the policy ACTUALLY outputs on each head
(if |dw| ~ 0 the policy already ignores it -> answered without ablation).

Run:  python ablate_heads.py [checkpoint.pt]
"""

import sys
import numpy as np
import torch

from residual_marl_env import ResidualMARLEnv
from networks import Actor
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS

CKPT = sys.argv[1] if len(sys.argv) > 1 else "residual_mappo.pt"


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(obs_dim=ck["obs_dim"], act_dim=ck["act_dim"])
    actor.load_state_dict(ck["state_dict"]); actor.eval()
    return actor, ck["obs_mean"].astype(np.float32), ck["obs_std"].astype(np.float32), ck["act_dim"]


def rollout(env, actor, om, os_, n, mode):
    """One deterministic held-out episode. mode in {full,no_dw,no_dlam,base}.
    Returns eval metrics + mean |dlam|,|dw| of the RAW policy output (pre-ablation)."""
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
    obs, _ = env.reset(seed=EVAL_SEED)
    agents = env.possible_agents
    floor = env.epsilon + env.stall_margin
    loops, loads, swings, rews, speeds = [], [], [], [], []
    mag_lam, mag_w = [], []
    k = 0
    while env.agents:
        oa = np.stack([obs[a] for a in agents]).astype(np.float32)
        with torch.no_grad():
            mean = actor(torch.tensor(((oa - om) / os_).astype(np.float32))).numpy()
        mag_lam.append(np.mean(np.linalg.norm(mean[:, :n], axis=1)))       # raw |dlam| per drone
        mag_w.append(np.mean(np.linalg.norm(mean[:, n:n + 6], axis=1)))    # raw |dw| per drone
        act = mean.copy()
        if mode in ("no_dw", "base"):
            act[:, n:n + 6] = 0.0
        if mode in ("no_dlam", "base"):
            act[:, :n] = 0.0
        obs, rewards, _, _, infos = env.step({a: act[i] for i, a in enumerate(agents)})
        rews.append(np.mean([rewards[a] for a in agents]))
        loops.append(np.mean([infos[a]["loop_dist"] for a in agents]))
        loads.append(infos[agents[0]]["load_err"])
        swings.append(infos[agents[0]]["load_verr"])
        k += 1
        if k > env.stall_grace:
            speeds.append(infos[agents[0]]["min_speed"])
    speeds = np.asarray(speeds)
    return dict(reward=float(np.mean(rews)), loop=float(np.mean(loops)),
                load=float(np.mean(loads)), loadmax=float(np.max(loads)),
                swing=float(np.mean(swings)), vmin=float(speeds.min()),
                stallfrac=float((speeds < floor).mean()),
                mag_lam=float(np.mean(mag_lam)), mag_w=float(np.mean(mag_w)))


if __name__ == "__main__":
    actor, om, os_, act_dim = load_policy(CKPT)
    env = ResidualMARLEnv(**DESYNC)
    n = env.n
    print(f"checkpoint: {CKPT}   act_dim={act_dim}  (n={n} -> dlam[:{n}], dw[{n}:{n+6}])")
    print(f"held-out eval: seed={EVAL_SEED} delays={EVAL_DELAYS}\n")

    # magnitudes are the same across modes (raw output) -> grab from the full run
    full = rollout(env, actor, om, os_, n, "full")
    print(f"policy OUTPUT magnitude (raw, pre-clip):  mean|dlam| {full['mag_lam']:.4f}   "
          f"mean|dw| {full['mag_w']:.4f}   ratio dw/dlam {full['mag_w']/max(full['mag_lam'],1e-9):.2f}\n")

    print(f"{'mode':9s} {'DET_R':>8s} {'loop':>7s} {'load':>7s} {'loadmax':>8s} "
          f"{'swing':>7s} {'vmin':>7s} {'stall%':>7s}")
    for mode in ("full", "no_dw", "no_dlam", "base"):
        e = full if mode == "full" else rollout(env, actor, om, os_, n, mode)
        print(f"{mode:9s} {e['reward']:8.3f} {e['loop']:7.3f} {e['load']:7.3f} {e['loadmax']:8.3f} "
              f"{e['swing']:7.3f} {e['vmin']:7.3f} {100*e['stallfrac']:7.1f}")
    env.close()
