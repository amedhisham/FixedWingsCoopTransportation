"""
deploy_compare.py — NET vs OPTIMIZER on ONE held-out trajectory (overlay).

deploy_prdot stays net-only (fast, no CasADi). This script is the slow, side-by-side
sanity check: it runs the DAgger'd pR_dot net (with the in-loop EMA, exactly as deployed)
AND the classical optimizer expert on the SAME held-out reference, then overlays:
  1. drone velocity norms   (net solid vs optimizer dashed, + epsilon)
  2. lambda per drone       (net vs optimizer)
  3. drone XY trajectories  (net vs optimizer, + load paths)

Run:  python deploy_compare.py         # HELD_IDX-th held-out trajectory
"""

import numpy as np
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from optimizer import cable_force_calculation
from controller import error_calculation, get_reference_trajectory
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from deploy_prdot import load_policy, run_episode, POLICY
from trajectories import heldout_set, custom_set, showcase_set

# SHOWCASE: net-vs-optimizer overlay on ONE demo trajectory (decoupled from the training set) at
# its OWN horizon — shows the net handles any length. None | "short" | "long" (trajectories.
# showcase_set); SHOWCASE_IDX picks which entry (0=line, 1..=quintics). Overrides the toggles below.
SHOWCASE = "short"
SHOWCASE_IDX = 1      # 0 -> the straight-line demo; 1.. -> the quintic demos
SHOWCASE_M = 3        # quintics available per preset (so SHOWCASE_IDX can reach 1..M)

# next 3 lines kinda dead code 
HELD_IDX = 0          # which held-out trajectory to compare on (0-based)
USE_DEFAULT = False    # True -> the original straight-line DEFAULT trajectory (traj=None, harshest)
USE_CUSTOM = None    # set to a custom index (0..4: +x,+y,+x+y,-x+y,+x-y) to survey a solver-engaging
                      #   custom traj instead (overrides USE_DEFAULT). None -> off.

BYPASS_OPT = False     # adaptive optimizer (the real expert), matches collection
RUN_NET = True         # False -> optimizer ONLY (skip the net, fast) to survey the optimizer per traj


def run_episode_opt(env, agent, Bb, L0, traj=None, t_end=None):
    """Closed-loop OPTIMIZER episode on `traj` (mirrors collect_prdot_data's expert rollout).
    Returns the SAME dict shape as deploy_prdot.run_episode so plotting is symmetric.
    `t_end` overrides the loop length (default T_END) — used for the compact SHOWCASE."""
    t_end = T_END if t_end is None else t_end
    obs42, _ = env.reset()
    agent.reset()
    prev_f = np.array([0.0, 0.0, FZ] * N)
    t_hist, load_hist, ref_hist = [], [], []
    dpos = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]            # PLANT drone speed (from the FMU)
    vRi_hist = [[] for _ in range(N)]        # optimizer's ANALYTIC ||v_Ri|| (the epsilon-constraint value)
    lam_hist = [[] for _ in range(N)]
    A_hist, xi_hist, solved_hist = [], [], []  # optimizer decision vars + whether the solver engaged

    t = 0.0
    while t < t_end - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t, traj)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)
        lam, _ = agent.optimize(t, R, vel, angvel, w_d, bypass=BYPASS_OPT)   # the expert lambda
        xi_hist.append(float(agent.prev_x[0])); A_hist.append(float(agent.prev_x[1]))  # [xi, A]
        solved_hist.append(bool(agent.solved))
        f_full, _ = cable_force_calculation(R, Bb, w_d, lam, N)

        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()

        t_hist.append(t)
        load_hist.append(pos.copy())
        ref_hist.append(get_reference_trajectory(t, traj)[0].copy())
        for i in range(N):
            dpos[i].append(obs42[18 + 3 * i: 18 + 3 * i + 3].copy())
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
            vRi_hist[i].append(float(agent.last_vRi[i]))
            lam_hist[i].append(lam[i])
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT

    load = np.array(load_hist); ref = np.array(ref_hist)
    err = np.linalg.norm(load - ref, axis=1)
    dvel = [np.array(v) for v in dvel]
    return dict(t=np.array(t_hist), load=load, ref=ref,
                dpos=[np.array(p) for p in dpos], dvel=dvel,
                vRi=[np.array(v) for v in vRi_hist],
                lam=[np.array(l) for l in lam_hist],
                A=np.array(A_hist), xi=np.array(xi_hist), solved=np.array(solved_hist),
                track_mean=float(err.mean()), track_max=float(err.max()),
                vmin=float(min(v.min() for v in dvel)))


def plot_compare(net, opt, title=""):
    """Overlay net (solid) vs optimizer (dashed). net may be None (optimizer-only survey)."""
    t = opt["t"]; sfx = f" — {title}" if title else ""
    have_net = net is not None

    # 1. drone velocity norms: PLANT speed (net solid, opt dashed) + epsilon floor.
    plt.figure(figsize=(11, 5.5))
    for i in range(N):
        if have_net:
            plt.plot(t, net["dvel"][i], color=f"C{i}", lw=1.4)
        plt.plot(t, opt["dvel"][i], color=f"C{i}", lw=1.2, ls="--", alpha=0.7)
    plt.axhline(EPS, c="gray", ls="--", lw=1.3, label="epsilon (constraint floor)")
    if have_net:
        plt.plot([], [], "k-", lw=1.4, label="plant speed — net")
    plt.plot([], [], "k--", alpha=0.7, label="plant speed — optimizer")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title("Drone velocity norms — plant speed" + sfx)
    plt.legend(fontsize=8); plt.grid(True)

    # 2. lambda per drone
    fig, ax = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
    for i in range(N):
        if have_net:
            ax[i].plot(t, net["lam"][i], color="C0", lw=1.2, label="net")
        ax[i].plot(t, opt["lam"][i], color="k", lw=1.0, ls="--", alpha=0.8, label="optimizer")
        ax[i].set_ylabel(f"$\\lambda_{i+1}$"); ax[i].grid(True)
        if i == 0:
            ax[i].legend(loc="upper right")
    ax[-1].set_xlabel("Time (s)"); fig.suptitle("Lambda — net vs optimizer" + sfx)

    # 3. drone XY trajectories + load
    plt.figure(figsize=(8, 7))
    for i in range(N):
        if have_net:
            plt.plot(net["dpos"][i][:, 0], net["dpos"][i][:, 1], color=f"C{i}", lw=1.4,
                     label=f"drone {i+1} net")
        plt.plot(opt["dpos"][i][:, 0], opt["dpos"][i][:, 1], color=f"C{i}", lw=1.2, ls="--",
                 alpha=0.7, label=f"drone {i+1} opt")
    if have_net:
        plt.plot(net["load"][:, 0], net["load"][:, 1], "k", lw=2, label="load net")
    plt.plot(opt["load"][:, 0], opt["load"][:, 1], color="gray", lw=2, ls="--", label="load opt")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("Drone XY trajectories — net (solid) vs optimizer (dashed)" + sfx)
    plt.legend(ncol=2, fontsize=8); plt.grid(True); plt.axis("equal")

    # 4. optimizer decision variables A (amplitude) and xi (frequency) — optimizer only
    if "A" in opt:
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(t, opt["A"], "C3"); ax[0].set_ylabel("A (amplitude)"); ax[0].grid(True)
        ax[1].plot(t, opt["xi"], "C4"); ax[1].set_ylabel("$\\xi$ (frequency)"); ax[1].grid(True)
        ax[1].set_xlabel("Time (s)")
        fig.suptitle("Optimizer decision variables  ($\\lambda_i = A\\cos(\\xi t + \\phi_i)$)" + sfx)


def main():
    t_end = T_END
    title = f"held-out #{HELD_IDX}"
    if SHOWCASE is not None:
        label, traj, t_end = showcase_set(SHOWCASE, SHOWCASE_M)[SHOWCASE_IDX]
        title = f"{label}  (t_end={t_end:.0f}s)"
        print(f"SHOWCASE '{SHOWCASE}' [{SHOWCASE_IDX}] -> {label}   t_end={t_end:.0f}s\n")
    elif USE_CUSTOM is not None:
        traj, p = custom_set()[USE_CUSTOM]
        title = f"custom {p['name']}"
        print(f"CUSTOM solver-engaging trajectory '{p['name']}' "
              f"(vel*move = dpos {np.round(p['pos_delta'], 2)}, move [5,30]s)\n")
    elif USE_DEFAULT:
        traj = None
        title = "default straight-line"
        print("DEFAULT straight-line trajectory (traj=None, v=1.1 — harshest)\n")
    else:
        traj, p = heldout_set(HELD_IDX + 1)[HELD_IDX]
        print(f"held-out #{HELD_IDX}: dpos={np.round(p['pos_delta'], 2)} "
              f"drot(deg)={np.round(np.rad2deg(p['rot_delta']), 1)}\n")

    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=t_end)
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    net = None
    if RUN_NET:
        net_p, om, os_ = load_policy(POLICY)
        print("running net (with in-loop EMA) ...")
        net = run_episode(net_p, om, os_, env, agent, Bb, L0, traj, t_end=t_end)
    print("running optimizer (CasADi) ...")
    opt = run_episode_opt(env, agent, Bb, L0, traj, t_end=t_end)
    env.close()

    print(f"\n{'':11} {'track_mean':>10} {'track_max':>10} {'vmin':>7}")
    if net is not None:
        print(f"{'net':>11} {net['track_mean']:>10.4f} {net['track_max']:>10.4f} {net['vmin']:>7.3f}")
    print(f"{'optimizer':>11} {opt['track_mean']:>10.4f} {opt['track_max']:>10.4f} {opt['vmin']:>7.3f}")

    # Deadbanded optimizer: does it ENGAGE (solve) or hold A/xi flat on this trajectory?
    print(f"solver engaged: {100*float(np.mean(opt['solved'])):.1f}% of steps  (deadband {agent.solve_below})")

    # WHY do A/xi ramp? cost has no upward term (w_pos=w_vel=0), so only a BINDING eps-constraint
    # can. Test: does min-over-drones analytic v_Ri touch eps, and is the solver pinned to +delta?
    vRi_closest = min(opt["vRi"][i].min() for i in range(N))
    dxi, dA = np.diff(opt["xi"]), np.diff(opt["A"])
    pin_xi = float(np.mean(dxi > 0.9 * agent.delta_xi))    # steps at (or near) the upper bound
    pin_A = float(np.mean(dA > 0.9 * agent.delta_A))
    print(f"\nanalytic v_Ri closest approach to eps: {vRi_closest:.3f}  (eps={EPS} -> "
          f"{'BINDS' if vRi_closest <= EPS + 1e-3 else 'slack, never binds'})")
    print(f"solver pinned to UPPER bound (+delta): xi {100*pin_xi:.0f}% of steps, A {100*pin_A:.0f}%")
    print(f"xi {opt['xi'][0]:.2f} -> {opt['xi'][-1]:.2f}   A {opt['A'][0]:.2f} -> {opt['A'][-1]:.2f}")

    plot_compare(net, opt, title=title)
    plt.show()


if __name__ == "__main__":
    main()
