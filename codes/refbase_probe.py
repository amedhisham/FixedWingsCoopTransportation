"""
refbase_probe.py — does feeding the SHARED REFERENCE load state to each local agent fix coordination?

Follow-up to kf_probe.py (which killed the ESTIMATION route: per-drone filters diverge -> can't
manufacture agreement). The only signal all 4 drones hold IDENTICALLY with zero reconstruction is the
REFERENCE. This probe hands each drone's local replica the REFERENCE load state in place of its own
noisy/delayed view (same splice as kf_probe, overriding env._estimates). Every replica then runs its
NORMAL pipeline (w_d, G, N, lambda) on the SAME reference -> identical computation -> identical forces
-> internal forces cancel by construction. NOTE: this drives EVERYTHING off the reference, incl.
load-tracking w_d, so it is pure FEEDFORWARD (no actual-state feedback) -> expect coordination to
recover but tracking to suffer if the reference is a poor stand-in for the true state.

Base only, ZERO residual. Modes (loop deviation, lower=better):
  raw      real desync, each drone on its own noisy view   -> base_noisy (~1.1, broken)
  ref      each drone on the REFERENCE load state          -> THE thesis-relevant test
  perfect  each drone on the TRUE current load             -> base_clean (~0.013, ceiling)

Reading:
  ref ~ perfect       -> the reference CARRIES coordination -> a classical feedforward fixes it, NO RL
                         -> the RL-for-coordination thesis is at risk (better to know NOW).
  ref ~ raw           -> the reference does NOT help -> lambda is genuinely state-coupled -> RL needed.
  ref PARTIAL (coord drops but loop worse than perfect) -> feedforward coordinates the NOMINAL but is
                         RIGID (no nullspace disturbance rejection) -> RL owns the coord-AND-loop
                         frontier -> thesis SAFE.

Run:  python refbase_probe.py
"""

import numpy as np
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv
from controller import get_reference_trajectory
from collect_il_data import T_END
from expert_reference import training_pairs, eval_scenarios
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS, DISABLE_DW

TRAIN_Q_IDX_OFFSET = 5          # which training quintic to probe (same as diagnose_f2 / kf_probe)


def _override(env, obs42, mode):
    """Per-drone load state handed to the replicas, per MODE. Called AFTER env._update_estimates."""
    if mode == "raw":
        return env._estimates
    if mode == "perfect":
        p, R, v, w = env._unpack_load(obs42)
        return [(p.copy(), R.copy(), v.copy(), w.copy()) for _ in range(env.n)]
    if mode == "ref":                                          # the SHARED reference load state (same for all)
        pd, vd, Rd, wd = get_reference_trajectory(env.t, env.traj)
        Rd = np.asarray(Rd)
        return [(pd.copy(), Rd.copy(), vd.copy(), wd.copy()) for _ in range(env.n)]
    raise ValueError(mode)


def rollout(mode, traj, epos):
    """Base-only (ZERO residual) episode with the load view spliced per MODE. Deterministic."""
    env = ResidualMARLEnv(**DESYNC, disable_dw=DISABLE_DW, end_time=T_END)
    env.traj, env.expert_pos = traj, epos
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)

    orig = env._update_estimates
    def patched(obs42):
        orig(obs42)
        env._estimates = _override(env, obs42, mode)
    env._update_estimates = patched

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
    modes = ["raw", "ref", "perfect"]
    colors = {"raw": "C3", "ref": "C0", "perfect": "C2"}

    print(f"ref-base probe  desync {DESYNC}  delays {list(EVAL_DELAYS)}  (base only, zero residual)\n")
    print(f"{'scenario':<15} | {'metric':<6} | {'raw':>10} {'ref':>10} {'perfect':>10} |  ref closes")
    print("-" * 74)

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 9), sharex=True)
    for row, (label, traj, epos) in enumerate(scenarios):
        res = {m: rollout(m, traj, epos) for m in modes}
        raw_l, ref_l, pf_l = res["raw"]["loop"], res["ref"]["loop"], res["perfect"]["loop"]
        gap = raw_l - pf_l
        closed = 100.0 * (raw_l - ref_l) / gap if gap > 1e-9 else float("nan")
        for metric in ("loop", "load", "coord"):
            r, k, p = res["raw"][metric], res["ref"][metric], res["perfect"][metric]
            tail = f"{closed:>6.0f}% of loop gap" if metric == "loop" else ""
            print(f"{label:<15} | {metric:<6} | {r:>10.3f} {k:>10.3f} {p:>10.3f} |  {tail}")
        print("-" * 74)

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
    fig.suptitle("Reference load state -> base (base only, zero residual): does the shared reference coordinate?\n"
                 "raw = own noisy view | ref = shared REFERENCE state | perfect = true load (ceiling)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "refbase_probe_loop.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")
    print("\nverdict: ref~perfect -> reference carries coordination (classical fix, thesis risk).  "
          "ref~raw -> lambda state-coupled, RL needed.  "
          "ref partial (coord down, loop worse) -> RL owns coord-AND-loop frontier (thesis safe).")
    plt.show()


if __name__ == "__main__":
    main()
