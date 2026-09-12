"""interp_search.py — is the -0.392 -> -0.383 descent a STRAIGHT path the spiral took the slow way
(rotation tax), or did the winding route matter? INTERPOLATE between the two real endpoints and eval
DET_R along the line. No collection, just evals (~3 min, laptop). See [[f2-rotation-limited]].

  theta_A = LS_CKPT_A (start ~ -0.392)   theta_B = LS_CKPT_B (end ~ -0.383)
  d = theta_B - theta_A = the FULL spiral's net drift (free).  eval DET_R at theta_A + alpha*d.

Reading:
  alpha in [0,1] MONOTONE downhill, alpha=1 ~ -0.383  -> straight path exists, spiral = rotation tax (Lookahead wins)
  BUMP/BARRIER in [0,1]                               -> endpoints in different basins, winding was needed
  alpha > 1 keeps dropping                            -> can extrapolate PAST -0.383
Run:  PYTHONPATH=. python -u interp_search.py       (optionally LS_CKPT_A=... LS_CKPT_B=...)
Measured 2026-09-12: 0.00 -0.392 | 0.50 -0.383 | 0.75 -0.380 (BEST, beats both ends) | 1.0 -0.385 (=B) | 2.0 -0.413.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
import mappo
from mappo import (DESYNC, DISABLE_DW, CONSIST_W, CONSIST_LAM_W, OVERFIT_END, T_END, OVERFIT,
                   HIDDEN, DEVICE, SEED, eval_policy, overfit_set)
from residual_marl_env import ResidualMARLEnv
from networks import Actor

CKPT_A = os.environ.get("LS_CKPT_A", "residual_mappo_overfit_xyz_2trj_ch.pt")   # start (~ -0.392)
CKPT_B = os.environ.get("LS_CKPT_B", "residual_mappo_overfit.pt")               # end   (~ -0.383)
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def flat(sd, keys):
    return torch.cat([sd[k].reshape(-1).float() for k in keys])


def main():
    torch.manual_seed(SEED)
    end_time = OVERFIT_END if OVERFIT else T_END
    env = ResidualMARLEnv(**DESYNC, disable_dw=DISABLE_DW, end_time=end_time,
                          track_clean_lambda=(CONSIST_W > 0 or CONSIST_LAM_W > 0))
    env.reset(seed=SEED)
    env.default_expert_pos = env.expert_pos.copy()
    pairs, eval_scen = overfit_set()
    obs_dim = env._obs_space.shape[0]
    act_dim = env._act_space.shape[0]
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=HIDDEN).to(DEVICE)

    ckA = torch.load(CKPT_A, map_location=DEVICE, weights_only=False)
    ckB = torch.load(CKPT_B, map_location=DEVICE, weights_only=False)
    keys = list(actor.state_dict().keys())
    thetaA = flat(ckA["state_dict"], keys)
    thetaB = flat(ckB["state_dict"], keys)
    om = ckA["obs_mean"].astype(np.float32)
    os_ = ckA["obs_std"].astype(np.float32)
    d = thetaB - thetaA
    print(f"CKPT_A = {CKPT_A}")
    print(f"CKPT_B = {CKPT_B}")
    print(f"||theta_B - theta_A|| = {d.norm():.4e}\n", flush=True)

    def set_theta(vec):
        sd = actor.state_dict()
        i = 0
        for k in keys:
            n = sd[k].numel()
            sd[k].copy_(vec[i:i + n].view_as(sd[k]))
            i += n

    set_theta(thetaB)
    omB = ckB["obs_mean"].astype(np.float32); osB = ckB["obs_std"].astype(np.float32)
    eB = eval_policy(env, actor, omB, osB, eval_scen)
    print(f"[sanity] CKPT_B under its own norm: DET_R {eB['reward']:+.4f}  loop {eB['loop']:.4f}\n", flush=True)

    print("alpha   DET_R      loop     per-traj                        blow")
    for al in ALPHAS:
        set_theta(thetaA + al * d)
        e = eval_policy(env, actor, om, os_, eval_scen)
        pt = "  ".join(f"{k}:{v:.3f}" for k, v in e["per_traj_loop"].items())
        tag = "  <- CKPT_A (start)" if al == 0.0 else ("  <- CKPT_B (end)" if al == 1.0 else "")
        print(f"{al:5.2f}  {e['reward']:+.4f}   {e['loop']:.4f}   {pt}   {e['blowups']}{tag}", flush=True)


if __name__ == "__main__":
    main()
