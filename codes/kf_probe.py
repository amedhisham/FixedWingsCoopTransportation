"""
kf_probe.py — the thesis-deciding experiment: can a KF in front of the base fix coordination?

Desync corrupts ONLY each drone's LOAD VIEW. Physically: each drone reads an AprilTag on the load
with its OWN camera, from its OWN viewpoint, on a fast-moving system -> the four "load readings" are
genuinely DIFFERENT estimates, not noisy copies of one number. Modeled as DESYNC = pos_noise 0.03,
vel_noise 0.10, AR(1) corr 0.995 (a slow, bias-like, viewpoint-dependent drift), plus per-drone
delays [1,2,2,1] that RANDOM-WALK in [0, d_i] each step. This view feeds the drone's local replica
via env._estimates[i]. Clean coordination is emergent: identical replicas on the SAME view -> lambdas
agree -> internal forces cancel (base_clean loop ~0.013). Desync -> each replica a DIFFERENT view ->
lambdas disagree -> leak -> loop ~1.10.

THE TEST (no residual anywhere): per-drone KF in front of the base. Each drone independently filters
its own noisy+delayed stream and feeds the RECONSTRUCTED load to its replica.

  raw     = the real corrupted view   -> base           -> base_noisy (~1.10, broken)
  kf      = each drone's KF estimate  -> base           -> the answer
  perfect = the true current load     -> base           -> base_clean (~0.013, ceiling)

The KF here is DELIBERATELY GENEROUS -> a BEST-CASE upper bound on estimation:
  * R = the injected marginal std. NOT fair (the real camera/AprilTag error is non-white, non-stationary,
    viewpoint-dependent -> uncalibratable). A realistic KF would be WORSE. We grant it to bound the best case.
  * The KF assumes WHITE noise, but the injected noise is AR(1) 0.995 (a slow bias) -> it CANNOT remove
    the bias-like part. That residual is the whole point.
  * Delay is UNKNOWN and time-varying in [0,2], per-drone -> the KF cannot predict-ahead by the true lag.
    It applies a single BLIND constant guess (KF_LOOKAHEAD), the most a drone could honestly do.

  kf ~ raw     -> even the BEST-CASE KF can't clean it -> DECISIVE: estimation is not the lever,
                  the residual must be ROBUST CONTROL. (A realistic KF is only worse -> conclusion holds.)
  kf ~ perfect -> estimation could help, but this is the OPTIMISTIC ceiling; the real camera KF lands
                  worse -> an upper bound, not a green light.

Run:  python kf_probe.py
"""

import numpy as np
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv
from collect_il_data import T_END
from expert_reference import training_pairs, eval_scenarios
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS, DISABLE_DW

TRAIN_Q_IDX_OFFSET = 5          # which training quintic to probe (same as diagnose_f2)

# --- KF knobs. R from the injected std = GENEROUS/best-case (see docstring; not fair, granted to bound
#     the best case). jerk_std = process-noise (load smoothness) knob. LOOKAHEAD = a BLIND fixed guess of
#     the delay (the true per-drone, time-varying [0,2] lag is UNKNOWN -> a drone can only guess). ---
KF_POS_STD = 0.03
KF_VEL_STD = 0.10
KF_JERK_STD = 6.0
KF_LOOKAHEAD = 1                 # blind delay guess (steps); the true lag is unknown + varies 0..2 per drone


class LoadKF:
    """Per-drone constant-acceleration Kalman filter on the load's LINEAR state (pos, vel) — the only
    noisy channels (rot/angvel noise are 0). State x=[p(3),v(3),a(3)]. De-noises pos/vel and extrapolates
    a BLIND fixed guess of the (unknown) delay. R/angvel are passed through (clean-but-delayed)."""

    def __init__(self, dt, pos_std, vel_std, jerk_std):
        self.dt = dt
        self.x = None
        self.P = np.eye(9)
        self.H = np.zeros((6, 9))
        self.H[0:3, 0:3] = np.eye(3)
        self.H[3:6, 3:6] = np.eye(3)
        self.R = np.diag([pos_std ** 2] * 3 + [vel_std ** 2] * 3)
        self.F = np.eye(9)
        self.F[0:3, 3:6] = dt * np.eye(3)
        self.F[0:3, 6:9] = 0.5 * dt ** 2 * np.eye(3)
        self.F[3:6, 6:9] = dt * np.eye(3)
        self.Q = self._white_jerk_Q(dt) * jerk_std ** 2

    @staticmethod
    def _white_jerk_Q(dt):
        """Discrete white-jerk process-noise (per axis), interleaved to state order [p3, v3, a3]."""
        b = np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                      [dt ** 4 / 8,  dt ** 3 / 3, dt ** 2 / 2],
                      [dt ** 3 / 6,  dt ** 2 / 2, dt]])
        Q = np.zeros((9, 9))
        for k in range(3):
            idx = [k, 3 + k, 6 + k]
            Q[np.ix_(idx, idx)] = b
        return Q

    def step(self, z_p, z_v, lookahead):
        """Fuse one noisy measurement, then extrapolate `lookahead` (blind guess) steps. Returns
        (p_hat, v_hat). Filter state advances one step (the measurement cadence)."""
        z = np.concatenate([z_p, z_v])
        if self.x is None:
            self.x = np.concatenate([z_p, z_v, np.zeros(3)])
        self.x = self.F @ self.x                                   # predict
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x                                    # update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P
        xa = self.x.copy()                                         # predict-ahead by the blind guess
        for _ in range(int(lookahead)):
            xa = self.F @ xa
        return xa[0:3], xa[3:6]


def _override(env, obs42, mode, kfs):
    """Per-drone load estimate handed to the replicas, per MODE. Called AFTER env._update_estimates
    (so env._estimates = RAW noisy/delayed view; env._noise_*/_delay_cur are set)."""
    if mode == "raw":
        return env._estimates
    p_true, R_true, v_true, w_true = env._unpack_load(obs42)       # current TRUE load
    if mode == "perfect":
        return [(p_true.copy(), R_true.copy(), v_true.copy(), w_true.copy()) for _ in range(env.n)]
    if mode == "kf":
        out = []
        for i in range(env.n):
            p_m, R_m, v_m, w_m = env._estimates[i]                 # raw noisy+delayed measurement
            p_hat, v_hat = kfs[i].step(p_m, v_m, KF_LOOKAHEAD)     # de-noise + BLIND lookahead (no true delay)
            out.append((p_hat, R_m, v_hat, w_m))                   # R/angvel clean-but-delayed -> pass through
        return out
    raise ValueError(mode)


def rollout(mode, traj, epos):
    """Base-only (ZERO residual) episode with the load view spliced per MODE. Returns per-step mean
    loop deviation + summary means. Deterministic (EVAL_SEED / EVAL_DELAYS)."""
    env = ResidualMARLEnv(**DESYNC, disable_dw=DISABLE_DW, end_time=T_END)
    env.traj, env.expert_pos = traj, epos
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
    kfs = [LoadKF(env.dt, KF_POS_STD, KF_VEL_STD, KF_JERK_STD) for _ in range(env.n)]

    orig = env._update_estimates
    def patched(obs42):
        orig(obs42)
        env._estimates = _override(env, obs42, mode, kfs)
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
    modes = ["raw", "kf", "perfect"]
    colors = {"raw": "C3", "kf": "C0", "perfect": "C2"}

    print(f"KF probe  desync {DESYNC}  delays {list(EVAL_DELAYS)}  "
          f"(GENEROUS KF: R from std, blind lookahead {KF_LOOKAHEAD}, jerk_std {KF_JERK_STD})\n")
    print(f"{'scenario':<15} | {'metric':<6} | {'raw':>10} {'kf':>10} {'perfect':>10} |  KF closes")
    print("-" * 72)

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 9), sharex=True)
    for row, (label, traj, epos) in enumerate(scenarios):
        res = {m: rollout(m, traj, epos) for m in modes}
        raw_l, kf_l, pf_l = res["raw"]["loop"], res["kf"]["loop"], res["perfect"]["loop"]
        gap = raw_l - pf_l
        closed = 100.0 * (raw_l - kf_l) / gap if gap > 1e-9 else float("nan")
        for metric in ("loop", "load", "coord"):
            r, k, p = res["raw"][metric], res["kf"][metric], res["perfect"][metric]
            tail = f"{closed:>6.0f}% of loop gap" if metric == "loop" else ""
            print(f"{label:<15} | {metric:<6} | {r:>10.3f} {k:>10.3f} {p:>10.3f} |  {tail}")
        print("-" * 72)

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
    fig.suptitle("KF-in-front-of-base: loop deviation over time (base only, zero residual)\n"
                 "raw = corrupted view | kf = GENEROUS filtered view | perfect = true load (ceiling)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "kf_probe_loop.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")
    print("\nverdict: kf~raw -> even best-case estimation fails -> residual = robust CONTROL (thesis safe).  "
          "kf~perfect -> estimation could help, but this KF is optimistic (real camera KF is worse).")
    plt.show()


if __name__ == "__main__":
    main()
