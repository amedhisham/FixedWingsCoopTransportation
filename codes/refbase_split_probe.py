"""
refbase_split_probe.py — the SURGICAL feedforward/feedback split (isolates lambda-from-reference).

refbase_probe.py drove EVERYTHING off the reference -> coord went to 0 (reference coordinates!) but
tracking EXPLODED, because w_d = PID(ref-ref) ~ 0-error removed all load feedback (open loop). That
conflated two things. This probe SPLITS them:
  * LOAD TRACKING (G+ w_d) + geometry (G, N): kept on each drone's REAL noisy measurement (estimates
    NOT overridden -> prepare() stashes the measured w_d/G/N -> feedback preserved).
  * COORDINATION (lambda, the N lambda part): computed from a SINGLE SHARED source and handed to all
    drones (env.net is replaced to return one shared lambda vector, broadcast).
So f_i = G+_i w_d_i (MEASURED, tracks the load) + N_i lambda_shared (SHARED, coordinates). This is the
2-DOF split: feedback tracking on the measurement, feedforward coordination on the shared signal.

Base only, ZERO residual. Modes (loop deviation, lower=better):
  raw          per-drone lambda, per-drone view                 -> base_noisy (~1.1, broken)
  ref_lambda   lambda from the REFERENCE, w_d MEASURED           -> THE thesis-relevant test
  true_lambda  lambda from the TRUE load, w_d MEASURED (diag)    -> best-case shared lambda + real
                                                                    tracking; ref_lambda vs true_lambda
                                                                    isolates the reference-as-proxy gap
  perfect      true load everywhere                              -> base_clean (~0.013, ceiling)

Reading (the question is: does ref_lambda TRACK while coord stays ~0?):
  ref_lambda: coord~0 AND loop small (~perfect) -> the reference COORDINATES AND TRACKS with no RL
                                                   -> classical competitor -> thesis at risk.
  ref_lambda: coord~0 but loop still BAD          -> reference lambda coordinates but can't track (it's
                                                   feedforward, no nullspace disturbance rejection) ->
                                                   RL owns the coord-AND-loop frontier -> thesis SAFE.
  ref_lambda bad, true_lambda good                -> reference is a poor lambda PROXY (needs the state)
                                                   -> RL bridges ref->true -> thesis SAFE.

Run:  python refbase_split_probe.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv, LocalModelAgent
from controller import get_reference_trajectory
from collect_il_data import T_END
from expert_reference import training_pairs, eval_scenarios
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS, DISABLE_DW

TRAIN_Q_IDX_OFFSET = 5


def _shared_lambda_net(env, source):
    """Replace env.net: IGNORE the per-drone (measured) rows, compute ONE lambda from a SHARED source
    (the reference, or the true load), broadcast to all n drones. A dedicated stateful replica carries
    its own reconstruction history. The drones' MEASURED w_d/G/N (from their own prepare) are untouched
    -> load tracking stays on the measurement; only lambda (coordination) rides the shared signal."""
    real_net = env.net
    shared = LocalModelAgent(env.n, env.dt, env.phases, env.epsilon, env.L0,
                             env.mass, env.J, env.Bb, env.recon_alpha)
    shared.reset()

    def net(_Xn_ignored):
        t = env.t
        if source == "ref":
            pd, vd, Rd, wd = get_reference_trajectory(t, env.traj)
            p, v, R, w = pd, vd, np.asarray(Rd), wd
        else:                                                  # true (diagnostic)
            p, R, v, w = env._unpack_load(env._obs42)
        row = shared.prepare(p, v, R, w, t, env.traj)
        Xr = ((row[None, :] - env.obs_mean) / env.obs_std).astype(np.float32)
        with torch.no_grad():
            lam = real_net(torch.tensor(Xr)).numpy()[0]        # (n,) shared lambda
        shared.finalize(lam.copy())
        return torch.tensor(np.tile(lam, (env.n, 1)).astype(np.float32))

    return net


def rollout(mode, traj, epos):
    env = ResidualMARLEnv(**DESYNC, disable_dw=DISABLE_DW, end_time=T_END)
    env.traj, env.expert_pos = traj, epos
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)

    if mode == "ref_lambda":
        env.net = _shared_lambda_net(env, "ref")               # lambda from reference; w_d/G/N stay MEASURED
    elif mode == "true_lambda":
        env.net = _shared_lambda_net(env, "true")
    elif mode == "perfect":
        orig = env._update_estimates
        def patched(obs42):
            orig(obs42)
            p, R, v, w = env._unpack_load(obs42)
            env._estimates = [(p.copy(), R.copy(), v.copy(), w.copy()) for _ in range(env.n)]
        env._update_estimates = patched
    # raw -> no patch

    env.reset(seed=EVAL_SEED)
    agents = env.possible_agents
    ad = env._act_space.shape[0]
    zero = {a: np.zeros(ad, np.float32) for a in agents}
    loops, loads, coords, blew = [], [], [], False
    while env.agents:
        _, _, _, _, infos = env.step(zero)
        if infos[agents[0]].get("blowup"):
            blew = True
            break
        loops.append(np.mean([infos[a]["loop_dist"] for a in agents]))
        loads.append(infos[agents[0]]["load_err"])
        coords.append(infos[agents[0]]["coord"])
    env.close()
    loops = np.asarray(loops)
    return dict(loop_ts=loops,
                loop=float(loops.mean()) if loops.size else float("nan"),
                load=float(np.mean(loads)) if loads else float("nan"),
                coord=float(np.mean(coords)) if coords else float("nan"),
                blew=blew)


def main():
    pairs, n_anchor = training_pairs()
    line_s = eval_scenarios()[0]
    held_s = eval_scenarios()[1]
    ti = n_anchor + TRAIN_Q_IDX_OFFSET
    train_s = (f"train_q{ti}", pairs[ti][0], pairs[ti][1])
    scenarios = [line_s, train_s, held_s]
    modes = ["raw", "ref_lambda", "true_lambda", "perfect"]
    colors = {"raw": "C3", "ref_lambda": "C0", "true_lambda": "C1", "perfect": "C2"}

    print(f"ref-base SPLIT probe  desync {DESYNC}  delays {list(EVAL_DELAYS)}  "
          f"(base only, zero residual; w_d MEASURED, lambda SHARED)\n")
    print(f"{'scenario':<15} | {'metric':<6} | {'raw':>9} {'ref_lambda':>11} {'true_lambda':>12} {'perfect':>9}")
    print("-" * 78)

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 9), sharex=True)
    for row, (label, traj, epos) in enumerate(scenarios):
        res = {m: rollout(m, traj, epos) for m in modes}
        for metric in ("loop", "load", "coord"):
            vals = "  ".join(f"{res[m][metric]:>9.3f}" for m in modes)
            print(f"{label:<15} | {metric:<6} | {vals}")
        print("-" * 78)

        ax = axes[row]
        for m in modes:
            ts = res[m]["loop_ts"]
            t = np.arange(ts.size) * 0.01
            lbl = f"{m}  (mean {res[m]['loop']:.3f})" + ("  BLEW" if res[m]["blew"] else "")
            ax.plot(t, ts, color=colors[m], lw=1.3, label=lbl)
        ax.set_yscale("log")
        ax.set_ylabel("loop dev (m)")
        ax.set_title(f"{label}", loc="left", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("SURGICAL split: w_d MEASURED (tracking) + lambda SHARED (coordination), base only\n"
                 "raw | ref_lambda = lambda from REFERENCE | true_lambda = lambda from true load | perfect",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "refbase_split_probe_loop.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")
    print("\nverdict: ref_lambda coord~0 AND loop small -> classical competitor (thesis risk).  "
          "ref_lambda coord~0 but loop still bad -> feedforward can't track -> RL owns the frontier (safe).  "
          "ref_lambda bad, true_lambda good -> reference is a poor lambda proxy -> RL bridges (safe).")
    plt.show()


if __name__ == "__main__":
    main()
