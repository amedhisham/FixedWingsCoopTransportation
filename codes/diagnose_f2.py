"""
diagnose_f2.py — localize the F2 tracking ceiling from a TRAINED checkpoint (no training loop).

Now a 3-WAY decomposition per scenario, so 0.33 finally has a DENOMINATOR:
  base_clean  = F1 base, ZERO residual, NO noise/delay   -> can the BASE even track this trajectory?
  base_noisy  = F1 base, ZERO residual, WITH desync      -> how much does desync HURT before the residual?
  residual    = the trained actor, WITH desync           -> how much does the residual RECOVER?
Also reports LOAD error per mode (base_noisy vs residual load -> does the residual HURT the load? nullspace leak)
and residual CAP SATURATION (sat = raw head norm / cap; >=1 => clipped, authority-limited).

Reading:
  base_clean ~ base_noisy ~ residual   -> the BASE can't track it even clean -> F1 problem, NOT F2.
  base_noisy >> residual > base_clean  -> residual recovers a lot but not all -> residual is the wall (memory lever).
  residual ~ base_noisy                -> residual barely helps -> residual CLASS is the wall (memory lever).

Run:  python diagnose_f2.py                       (default residual_mappo.pt)
      python diagnose_f2.py residual_mappo_gt1.pt
"""

import sys
import numpy as np
import torch

from residual_marl_env import ResidualMARLEnv
from networks import Actor
from expert_reference import training_pairs, eval_scenarios
from collect_il_data import T_END
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS, DISABLE_DW, DEVICE   # reuse the EXACT training/eval config

CKPT = sys.argv[1] if len(sys.argv) > 1 else "residual_mappo.pt"
TRAIN_Q_IDX_OFFSET = 5          # which training quintic to probe: pairs[n_anchor + this] (in-distribution quintic)


def load_actor(path):
    """Rebuild the Actor from a checkpoint, inferring hidden widths from the saved weight shapes."""
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ck["state_dict"]
    hidden = tuple(sd[k].shape[0] for k in sd if k.startswith("body.") and k.endswith(".weight"))
    actor = Actor(obs_dim=int(ck["obs_dim"]), act_dim=int(ck["act_dim"]), hidden=hidden).to(DEVICE)
    actor.load_state_dict(sd); actor.eval()
    om = ck["obs_mean"].astype(np.float32); os_ = ck["obs_std"].astype(np.float32)
    return actor, om, os_, hidden


def rollout(env, actor, om, os_, traj, epos, delays):
    """One DETERMINISTIC episode. actor=None -> ZERO residual (pure F1 base). Returns metrics dict."""
    env.traj, env.expert_pos = traj, epos
    env.ctrl_delay = np.asarray(delays, dtype=int)
    obs, _ = env.reset(seed=EVAL_SEED)
    agents = env.possible_agents
    ad = env._act_space.shape[0]
    loops, loads, coords, sat_lam, speeds = [], [], [], [], []
    blew = False
    while env.agents:
        if actor is None:
            act = {a: np.zeros(ad, np.float32) for a in agents}      # base only
        else:
            oa = np.stack([obs[a] for a in agents]).astype(np.float32)
            with torch.no_grad():
                mean = actor(torch.tensor(((oa - om) / os_).astype(np.float32), device=DEVICE)).cpu().numpy()
            act = {a: mean[i] for i, a in enumerate(agents)}
        obs, _, _, _, infos = env.step(act)
        if infos[agents[0]].get("blowup"):
            blew = True; break
        loops.append(np.mean([infos[a]["loop_dist"] for a in agents]))
        loads.append(infos[agents[0]]["load_err"])
        coords.append(infos[agents[0]]["coord"])            # ||sum f_int|| = net internal-force LEAK (load disturbance)
        sat_lam.append([infos[a]["sat_lam"] for a in agents])
        speeds.append(infos[agents[0]]["min_speed"])
    sat_lam = np.asarray(sat_lam)
    return dict(loop=float(np.mean(loops)) if loops else float("nan"),
                load=float(np.mean(loads)) if loads else float("nan"),
                coord=float(np.mean(coords)) if coords else float("nan"),
                sat_lam=float(sat_lam.mean()) if sat_lam.size else float("nan"),
                clip_lam=float((sat_lam >= 1.0).mean()) if sat_lam.size else float("nan"),
                blew=blew)


def main():
    actor, om, os_, hidden = load_actor(CKPT)
    env = ResidualMARLEnv(**DESYNC, disable_dw=DISABLE_DW, end_time=T_END)          # desync (training distribution)
    clean = {k: 0.0 for k in DESYNC}                                               # noise OFF
    env_clean = ResidualMARLEnv(**clean, disable_dw=DISABLE_DW, end_time=T_END)     # no noise + (below) no delay

    pairs, n_anchor = training_pairs()
    line_s, held_s = eval_scenarios()
    ti = n_anchor + TRAIN_Q_IDX_OFFSET
    train_s = (f"train_q{ti}", pairs[ti][0], pairs[ti][1])
    scenarios = [line_s, train_s, held_s]

    print(f"checkpoint {CKPT}  (actor hidden {hidden})   desync {DESYNC}  delays {list(EVAL_DELAYS)}\n")
    print(f"{'scenario':<15} | {'base_clean':>10} | {'base_noisy':>27} | {'residual':>27}")
    print(f"{'':<15} | {'loop':>10} | {'loop':>9} {'load':>9} {'coord':>7} | {'loop':>9} {'load':>9} {'coord':>7}")
    rows = {}
    for label, traj, epos in scenarios:
        bc = rollout(env_clean, None,  om, os_, traj, epos, [0, 0, 0, 0])   # base, no noise, no delay
        bn = rollout(env,       None,  om, os_, traj, epos, EVAL_DELAYS)    # base, desync
        rs = rollout(env,       actor, om, os_, traj, epos, EVAL_DELAYS)    # residual, desync
        rows[label] = (bc, bn, rs)
        print(f"{label:<15} | {bc['loop']:>10.3f} | {bn['loop']:>9.3f} {bn['load']:>9.3f} {bn['coord']:>7.3f} | "
              f"{rs['loop']:>9.3f} {rs['load']:>9.3f} {rs['coord']:>7.3f}"
              + ("  BLEW" if rs['blew'] else ""))
    env.close(); env_clean.close()

    print("\n--- decomposition (quintic = the hard case) ---")
    for label in (train_s[0], held_s[0]):
        bc, bn, rs = rows[label]
        hurt = bn["loop"] - bc["loop"]          # how much desync hurts the base
        recov = bn["loop"] - rs["loop"]         # how much the residual recovers
        pct = 100 * recov / hurt if hurt > 1e-6 else float("nan")
        load_cost = rs["load"] - bn["load"]     # how much the residual HURTS the load (nullspace leak)
        print(f"{label:<15} desync hurts base {bc['loop']:.3f}->{bn['loop']:.3f} (+{hurt:.3f});  "
              f"residual recovers {recov:+.3f} ({pct:.0f}% of the gap);  load cost {load_cost:+.3f}")

    print("\n--- load leak: MAGNITUDE vs FREQUENCY (why the PID handles RL's leak worse) ---")
    for label in (train_s[0], held_s[0]):
        bc, bn, rs = rows[label]
        cr = rs["coord"] / bn["coord"] if bn["coord"] > 1e-9 else float("nan")   # net-force leak ratio RL/base
        lr = rs["load"] / bn["load"] if bn["load"] > 1e-9 else float("nan")      # load-error ratio RL/base
        print(f"{label:<15} coord(leak) base {bn['coord']:.3f} -> RL {rs['coord']:.3f} (x{cr:.2f});  "
              f"load base {bn['load']:.3f} -> RL {rs['load']:.3f} (x{lr:.2f})")
    print("  if coord ratio ~ load ratio  -> MAGNITUDE (RL just injects a bigger leak).")
    print("  if coord ratio << load ratio -> FREQUENCY (similar-size leak, but faster -> PID rejects it less).")
    print("\nif base_clean ~ residual -> F1/base problem (not F2). "
          "if base_noisy >> residual > base_clean -> residual is the wall (memory lever).")


if __name__ == "__main__":
    main()
