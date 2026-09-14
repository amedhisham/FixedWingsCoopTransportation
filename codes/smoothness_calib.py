"""
smoothness_calib.py — calibrate the deadband thresholds for the lane + smoothness reward redesign
([[f2-smoothness-manifold-replacement]]). Rolls the DEMONSTRATED EXPERT PATH (smooth, the "good" class)
and the current POLICY (the jerky one with the 90-degree snaps) on the 3 benchmark directions and dumps
expert-vs-policy distributions of two per-step quantities:

  * ACCELERATION proxy  |Δ‖v‖| / dt   (m/s^2)   -> sets a_thresh  (deadband on speed change)
  * HEADING RATE        angle(v_t, v_{t-1}) / dt (rad/s) -> sets w_thresh (deadband on turn rate)

Set each deadband threshold on the SEPARATING LINE between the smooth expert (low) and the snappy policy
(high) — a threshold there frees the normal orbit/ramp and penalizes only the artifacts. Also reports the
policy's per-axis position excursion from the expert point as a ballpark for the lane half-width h.

The EXPERT is the precomputed demonstration (expert_path), differentiated for velocity — it can't blow up,
unlike the zero-residual base. The POLICY is the deterministic mean action, rolled under the eval desync.
Run locally like scale_test.py. Drop a benchmark scale if the policy blows up on it (calibration needs a
NON-blown rollout; the smoothness artifact appears wherever the policy flies, not only at the hard frontier).
"""
import numpy as np
import torch
from residual_marl_env import ResidualMARLEnv
from networks import Actor
from controller import make_quintic_pose
from expert_reference import expert_path
from trajectories import BASE_POS, HOLD
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS

CKPT = "residual_mappo_overfit.pt"        # policy to calibrate against
# (label, dir, scale_m, ramp_s) — the 3 benchmark trajectories (the original mix). Lower a scale if the
# policy blows up on it: calibration only needs non-blown rollouts of the policy's normal (if jerky) motion.
BENCH = [
    ("+x+y+z",   (1.0,  1.0,  0.5), 10.0, 29.0),
    ("+x+y-z",   (1.0,  1.0, -0.5), 10.0, 29.0),
    ("+.2x-y-z", (0.2, -1.0, -1.0), 18.0, 52.0),
]
GRACE = 20                                 # skip the first N steps (init transient) — same as scale_test
PCTS = [50, 90, 95, 99, 100]


def load_actor(env):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    om = ck["obs_mean"].astype(np.float32).reshape(-1); os_ = ck["obs_std"].astype(np.float32).reshape(-1)
    sd = ck["state_dict"]
    hidden = (sd["body.0.weight"].shape[0], sd["body.2.weight"].shape[0])   # INFER width
    actor = Actor(om.shape[0], env._act_space.shape[0], hidden=hidden)
    actor.load_state_dict(sd); actor.eval()
    return actor, om, os_


def roll_policy(env, actor, om, os_, traj, dpos):
    """Deterministic mean-policy rollout -> (positions (T,n,3), times (T,), blew)."""
    env.traj, env.expert_pos = traj, dpos
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
    obs, _ = env.reset(seed=EVAL_SEED)
    agents = env.possible_agents; n = env.n
    P, T, blew = [], [], False
    while env.agents:
        arr = np.stack([obs[a] for a in agents]).astype(np.float32)
        with torch.no_grad():
            mean = actor.distribution(torch.tensor((arr - om) / os_)).mean.numpy()
        obs, _, _, _, infos = env.step({a: mean[i] for i, a in enumerate(agents)})
        if infos[agents[0]].get("blowup"):
            blew = True
            break
        s = env.state()
        P.append(s[18:18 + 3 * n].reshape(n, 3).copy())
        T.append(env.t)
    return np.asarray(P), np.asarray(T), blew


def smooth_stats(pos, dt):
    """pos: (T,n,3) on a uniform dt grid -> (accel, turn) flat arrays pooled over drones+steps.
    accel = |Δ‖v‖|/dt (m/s^2);  turn = angle(v_t, v_{t-1})/dt (rad/s). v = finite-difference of pos."""
    if len(pos) < 3:
        return np.array([]), np.array([])
    v = np.diff(pos, axis=0) / dt                 # (T-1,n,3)
    sp = np.linalg.norm(v, axis=2)                # (T-1,n)
    accel = (np.abs(np.diff(sp, axis=0)) / dt).ravel()          # (T-2,n) -> flat
    a, b = v[:-1], v[1:]                          # (T-2,n,3)
    na = np.linalg.norm(a, axis=2); nb = np.linalg.norm(b, axis=2)
    good = (na > 1e-6) & (nb > 1e-6)
    cos = np.ones_like(na)
    cos[good] = np.clip(np.sum(a * b, axis=2)[good] / (na[good] * nb[good]), -1.0, 1.0)
    turn = (np.arccos(cos) / dt)[good].ravel()                  # rad/s
    return accel[np.isfinite(accel)], turn[np.isfinite(turn)]


def _pct(x):
    return "  ".join(f"{p:>3}:{np.percentile(x, p):8.3f}" for p in PCTS) if len(x) else "  (none)"


def _sep(good, bad):
    """Suggested deadband threshold = geometric-mean separating line between the expert's 99th pct (top of
    the smooth-normal band) and the policy's 90th pct (its high/snappy end), floored at the expert p99."""
    hi = max(float(np.percentile(good, 99)), 1e-6)
    bd = max(float(np.percentile(bad, 90)), hi)
    return float(np.sqrt(hi * bd))


def report(label, ea, et, pa, pt, exc=None):
    print(f"[{label}]")
    print(f"  accel |Δ‖v‖| (m/s^2)   " + "  ".join(f"p{p}" for p in PCTS))
    print(f"    expert  {_pct(ea)}")
    print(f"    policy  {_pct(pa)}")
    print(f"  heading rate (rad/s)")
    print(f"    expert  {_pct(et)}")
    print(f"    policy  {_pct(pt)}")
    if len(ea) and len(pa):
        print(f"  -> a_thresh ~ {_sep(ea, pa):.3f} m/s^2    w_thresh ~ {_sep(et, pt):.3f} rad/s")
    if exc is not None and len(exc):
        p = lambda ax, q: np.percentile(exc[:, ax], q)
        print(f"  lane |p-expert| per axis (m)  p90 x/y/z {p(0,90):.2f}/{p(1,90):.2f}/{p(2,90):.2f}"
              f"   p99 {p(0,99):.2f}/{p(1,99):.2f}/{p(2,99):.2f}   (ballpark for h)")
    print()


def main():
    print(f"smoothness calibration  ckpt={CKPT}  (expert = demonstrated path, policy = deterministic mean)\n")
    pool = {"ea": [], "et": [], "pa": [], "pt": []}
    for label, d, s, ramp in BENCH:
        end = HOLD + ramp + 1.0
        env = ResidualMARLEnv(**DESYNC, end_time=end)
        actor, om, os_ = load_actor(env)
        traj = make_quintic_pose(np.array(d, float) * s, np.zeros(3), ramp, HOLD, np.asarray(BASE_POS, float))
        dpos, _, _ = expert_path(traj, end)
        exp_pos = np.transpose(np.asarray(dpos), (1, 0, 2))     # (n,T,3) -> (T,n,3)
        dt_exp = end / (exp_pos.shape[0] - 1)
        ea, et = smooth_stats(exp_pos[GRACE:], dt_exp)

        pol_pos, tpol, blew = roll_policy(env, actor, om, os_, traj, dpos)
        exc = None
        if blew or len(pol_pos) < GRACE + 3:
            print(f"[{label}] policy BLEW UP (excluded) — expert-only\n")
            pa = pt = np.array([])
        else:
            dt_pol = float(np.median(np.diff(tpol)))
            pa, pt = smooth_stats(pol_pos[GRACE:], dt_pol)
            L = min(len(pol_pos), exp_pos.shape[0])                # per-axis excursion for h
            exc = np.abs(pol_pos[GRACE:L] - exp_pos[GRACE:L]).reshape(-1, 3)
        report(label, ea, et, pa, pt, exc)
        for k, arr in zip(pool, (ea, et, pa, pt)):
            if len(arr):
                pool[k].append(arr)
        env.close()

    cat = lambda k: np.concatenate(pool[k]) if pool[k] else np.array([])
    print("=" * 60)
    report("POOLED (all benchmark trajectories)", cat("ea"), cat("et"), cat("pa"), cat("pt"))


if __name__ == "__main__":
    main()
