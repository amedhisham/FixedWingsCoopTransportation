"""
examine_base.py — can the CLASSICAL BASE track richer trajectories?

Before we commit to a trajectory family for generalization, check what the base
controller (PID wrench, NO acceleration feedforward) can actually follow, noise-free
and residual-free. If the base can't track a shape, the residual RL has nothing to
ride on. Runs the same coherent rollout as expert_reference.py, but PATCHES
controller.get_reference_trajectory with candidate trajectories and reports load
tracking error (mean / max, and max DURING the maneuver i.e. after the initial hold).

Trajectories tested (all keep R=I, no load yaw, like the current setup):
  straight  - the current reference (sanity: should match today's ~cm tracking)
  fast      - straight but 2x speed (speed generalization)
  turn      - +x then +y right-angle corner (velocity STEP - base already handles these)
  arc_r5/r2 - constant-speed circular arcs (continuous curvature -> needs centripetal
              accel the reference never provides -> expect curvature-dependent LAG)
"""

import numpy as np
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ

T_HOLD = 5.0                       # initial hold (matches current get_reference_trajectory)
V = 1.1                            # base move speed (matches current v_move)
Z = 1.39                          # flight altitude (matches current z_hover)
LEVEL = (np.eye(3), np.zeros(3))   # R_Ld, omega_Ld — no load rotation, as today


def make_traj(kind):
    """Return a get_reference_trajectory(t)-compatible fn: t -> (p, v, R, omega)."""
    t_move = T_END - T_HOLD

    def straight(t):
        s = max(0.0, min(t - T_HOLD, t_move))
        return np.array([V * s, 0, Z]), np.array([V, 0, 0]) if T_HOLD < t < T_END else np.zeros(3), *LEVEL

    def fast(t):
        v = 2 * V
        s = max(0.0, min((t - T_HOLD), t_move)) * v
        return np.array([s, 0, Z]), np.array([v, 0, 0]) if T_HOLD < t < T_END else np.zeros(3), *LEVEL

    def turn(t):
        half = t_move / 2.0
        if t <= T_HOLD:
            return np.array([0, 0, Z]), np.zeros(3), *LEVEL
        tau = t - T_HOLD
        if tau <= half:                                   # leg 1: +x
            return np.array([V * tau, 0, Z]), np.array([V, 0, 0]), *LEVEL
        x = V * half; ty = min(tau - half, half)          # leg 2: +y from the corner
        return np.array([x, V * ty, Z]), np.array([0, V, 0]), *LEVEL

    def arc(radius):
        def f(t):
            if t <= T_HOLD:
                return np.array([0, 0, Z]), np.zeros(3), *LEVEL
            tau = min(t - T_HOLD, t_move)
            phi = V * tau / radius                          # swept angle
            p = np.array([radius * np.sin(phi), radius * (1 - np.cos(phi)), Z])
            v = np.array([V * np.cos(phi), V * np.sin(phi), 0.0])
            return p, v, *LEVEL
        return f

    def quintic(deltas, ramp=10.0):
        """Paper Fig.10 style: rest-to-rest 5th-order poly, pose moves by `deltas` over `ramp` s,
        then holds. Zero vel AND accel at both ends -> smooth, gentle (peak accel ~5.77*d/ramp^2)."""
        deltas = np.asarray(deltas, float)
        base = np.array([0.0, 0.0, Z])
        def f(t):
            if t <= T_HOLD:
                return base.copy(), np.zeros(3), *LEVEL
            u = min((t - T_HOLD) / ramp, 1.0)
            s = 10 * u**3 - 15 * u**4 + 6 * u**5             # position profile 0->1
            sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / ramp  # velocity profile (feedforward)
            return base + deltas * s, deltas * sd, *LEVEL
        return f

    return {"straight": straight, "fast": fast, "turn": turn,
            "arc_r5": arc(5.0), "arc_r2": arc(2.0),
            "quintic_1d": quintic([1.0, 0.0, 0.0]),          # paper: 1 m in 10 s, single axis
            "quintic_3d": quintic([1.0, 1.0, 1.0])}[kind]    # 3D diagonal, 1 m per axis in 10 s


def run_base(traj_fn):
    """Coherent (noise-free) base rollout, no residual. Returns load pos + ref pos + min drone speed.
    The trajectory is passed CLEANLY via compute_forces(..., traj=traj_fn) -- no module-global patch."""
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)
    prev_f = np.array([0.0, 0.0, FZ] * N)
    load, ref, vmin = [], [], []
    t = 0.0
    while t < T_END - 1e-9:
        pos = obs[0:3]
        R = np.round(obs[3:12].reshape((3, 3), order="C"), 6)
        vel, w = obs[12:15], obs[15:18]
        f, _, _ = agent.compute_forces(pos, vel, R, w, t, traj=traj_fn)
        ff = LLC_ALPHA * f + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        load.append(pos.copy()); ref.append(traj_fn(t)[0].copy())
        dv = [np.linalg.norm(obs[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]) for i in range(N)]
        vmin.append(min(dv))
        obs, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()
    return np.array(load), np.array(ref), np.array(vmin)


if __name__ == "__main__":
    hold_steps = int(T_HOLD / DT)
    print(f"T_END={T_END}  DT={DT}  hold={T_HOLD}s  V={V}   (error in m; 'maneuver' = after the hold)\n")
    print(f"{'traj':10s} {'mean err':>9s} {'max err':>9s} {'maneuver max':>13s} {'vmin':>7s}")
    results = {}
    for kind in ["straight", "fast", "turn", "arc_r5", "arc_r2", "quintic_1d", "quintic_3d"]:
        load, ref, vmin = run_base(make_traj(kind))
        err = np.linalg.norm(load - ref, axis=1)
        results[kind] = (load, ref, err)
        print(f"{kind:10s} {err.mean():9.4f} {err.max():9.4f} {err[hold_steps:].max():13.4f} {vmin.min():7.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for kind, (load, ref, err) in results.items():
        ax[0].plot(load[:, 0], load[:, 1], label=f"{kind} (load)")
        ax[0].plot(ref[:, 0], ref[:, 1], "--", alpha=0.4)
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)"); ax[0].axis("equal")
    ax[0].set_title("Load path (solid) vs reference (dashed)"); ax[0].legend(); ax[0].grid(True)
    for kind, (_, _, err) in results.items():
        ax[1].plot(np.arange(len(err)) * DT, err, label=kind)
    ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("||load - ref|| (m)")
    ax[1].set_title("Load tracking error"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout(); plt.show()
