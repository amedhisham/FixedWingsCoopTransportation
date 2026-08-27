"""
consist_cap_probe.py — NO training. Does the consistency target dlam* = lambda_clean - lambda_base
FIT inside the residual's authority cap? The env caps ||dlam|| at cap_lam*||lambda_base||, so the
correction fits iff  r = ||lambda_clean - lambda_base|| / ||lambda_base||  <=  cap_lam.

Rolls base-only under desync with track_clean_lambda=True, reads per-step lambda_base (each drone's
noisy-view vector) and lambda_clean (base net on the TRUE shared state = the coordinated target),
and reports the distribution of r + the fraction that FITS at several cap_lam values.
"""
import numpy as np
from residual_marl_env import ResidualMARLEnv
from optimizer import calculate_grasp_and_nullspace
from expert_reference import eval_scenarios
from collect_il_data import T_END
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS

CAPS = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
GRACE = 20                      # skip startup (lambda history warming from LAM0)


def run(env, scen):
    label, traj, epos = scen
    env.traj, env.expert_pos = traj, epos
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
    obs, _ = env.reset(seed=EVAL_SEED)
    agents = env.possible_agents
    zero = {a: np.zeros(env._act_space.shape[0], np.float32) for a in agents}
    r_lam, r_w = [], []
    fnull, fexpl, coord, gfn, ftot = [], [], [], [], []   # null-leak vs explicit-lambda vs coord-metric
    k = 0
    while env.agents:
        obs, *_ = env.step(zero)
        k += 1
        if k <= GRACE or env._lam_clean is None:
            continue
        for i in range(env.n):
            lb = env._lam_base[i]
            r_lam.append(np.linalg.norm(env._lam_clean - lb) / (np.linalg.norm(lb) + 1e-12))
            wb = env._wd_base[i]
            r_w.append(np.linalg.norm(env._wd_clean - wb) / (np.linalg.norm(wb) + 1e-12))
        # NULL-LEAK of the assembled RANGE commands: F = [f_g_i] (each drone's G+ w_d slice).
        # If the w_d agreed, F would be pure range (F_null=0). Disagreement -> F_null != 0 = the
        # internal force that SCATTERS the drones, invisible to the load (G F_null ~ 0) AND to coord.
        R0 = env._state_buffer[-1][1]                      # true rotation at decision time
        G = calculate_grasp_and_nullspace(R0, env.Bb, env.n)[0]     # 6 x 3n true grasp
        F = env._fg.flatten()                              # assembled range commands (3n,)
        Gp = np.linalg.pinv(G)
        F_null = F - Gp @ (G @ F)                          # null(G) component of the range patchwork
        fnull.append(np.linalg.norm(F_null))
        fexpl.append(np.linalg.norm(env._flam.flatten()))  # explicit lambda internal force magnitude
        coord.append(np.linalg.norm(env._net_fint))        # the coord METRIC (net internal force)
        gfn.append(np.linalg.norm(G @ F_null))             # load wrench from the leak -> should be ~0
        ftot.append(np.linalg.norm(F))
    diag = dict(fnull=np.array(fnull), fexpl=np.array(fexpl), coord=np.array(coord),
                gfn=np.array(gfn), ftot=np.array(ftot))
    return np.array(r_lam), np.array(r_w), label, diag


def report(name, r, cap_default):
    pcts = np.percentile(r, [50, 90, 95, 99])
    print(f"  {name}: r = ||target||/||base||  mean {r.mean():.3f}  median {pcts[0]:.3f}  "
          f"p90 {pcts[1]:.3f}  p95 {pcts[2]:.3f}  p99 {pcts[3]:.3f}  max {r.max():.3f}")
    fits = "  ".join(f"cap{c}:{100*np.mean(r <= c):.0f}%" for c in CAPS)
    print(f"    fits within cap (r<=cap): {fits}   [current cap={cap_default}]")


def main():
    env = ResidualMARLEnv(**DESYNC, end_time=T_END, track_clean_lambda=True)
    L, W = [], []
    print("reward-free probe: does the CLEAN-view consistency correction fit the residual caps?")
    print("  dlam* = lam_clean - lam_base (nullspace, cap_lam)   |   dw* = w_clean - w_base (range, cap_w)\n")
    for scen in eval_scenarios():
        r_lam, r_w, label, d = run(env, scen)
        L.append(r_lam); W.append(r_w)
        print(f"[{label}]")
        report("dlam*", r_lam, env.cap_lam)
        report("dw*  ", r_w, env.cap_w)
        print(f"  NULL-LEAK of range commands (the hidden scatter force):")
        print(f"    ||F_null|| (w_d-induced internal force) mean {d['fnull'].mean():.4f}   "
              f"as frac of ||F|| {100*(d['fnull']/d['ftot']).mean():.1f}%")
        print(f"    ||F_lambda|| (explicit lambda internal)  mean {d['fexpl'].mean():.4f}   "
              f"coord METRIC (net internal) mean {d['coord'].mean():.4f}")
        print(f"    ||G @ F_null|| (load wrench from the leak) mean {d['gfn'].mean():.2e}  <- ~0 => invisible to load")
        print(f"    => scatter force {d['fnull'].mean():.4f} is {d['fnull'].mean()/max(d['coord'].mean(),1e-9):.1f}x "
              f"the coord metric, and the load can't feel it\n")
    env.close()
    print("=== POOLED ===")
    report("dlam*", np.concatenate(L), env.cap_lam)
    report("dw*  ", np.concatenate(W), env.cap_w)


if __name__ == "__main__":
    main()
