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
from deploy_prdot import load_policy, run_episode, POLICY, euler_deg
from controller import make_quintic_pose
from trajectories import heldout_set, custom_set, showcase_set, BASE_POS, HOLD

# QUINTIC MOVE mode (like scale_test): dial a direction + magnitude + ramp and run a fresh rest-to-rest
# quintic, net vs optimizer. Overrides SHOWCASE/CUSTOM/DEFAULT when USE_MOVE is True. Horizon ends just
# after the move completes (HOLD + ramp + margin) — no long static-hold tail.
USE_MOVE = False
MOVE_DIR = (-0.06, -0.88, -0.47)   # movement direction (auto-normalized); scaled by MOVE_MAG
MOVE_MAG = 3.0               # displacement magnitude (m) along MOVE_DIR
MOVE_RAMP = 16.0             # quintic move duration (s)
MOVE_MARGIN = 2.0            # extra seconds after the move before the episode ends

# SHOWCASE: net-vs-optimizer overlay on ONE demo trajectory (decoupled from the training set) at
# its OWN horizon — shows the net handles any length. None | "short" | "long" (trajectories.
# showcase_set); SHOWCASE_IDX picks which entry (0=line, 1..=quintics). Overrides the toggles below.
SHOWCASE = "long"
SHOWCASE_IDX = 1      # 0 -> the straight-line demo; 1.. -> the quintic demos
SHOWCASE_M = 3        # quintics available per preset (so SHOWCASE_IDX can reach 1..M)

# next 3 lines kinda dead code 
HELD_IDX = 0          # which held-out trajectory to compare on (0-based)
USE_DEFAULT = False    # True -> the original straight-line DEFAULT trajectory (traj=None, harshest)
USE_CUSTOM = None    # set to a custom index (0..4: +x,+y,+x+y,-x+y,+x-y) to survey a solver-engaging
                      #   custom traj instead (overrides USE_DEFAULT). None -> off.

BYPASS_OPT = False     # adaptive optimizer (the real expert), matches collection
RUN_NET = True         # False -> optimizer ONLY (skip the net, fast) to survey the optimizer per traj
TIME_MARKS = 7         # 3-D trajectory plot: this many evenly-spaced "t=Xs" numbers along each DRONE path
                       #   so the spatial curves carry a time reference (0 -> off)


def move_descr(traj, t_end):
    """Describe a trajectory by its net MOVE — direction (unit) · magnitude (m) — from the reference's
    start->end displacement, so the title reads e.g. 'move [0.71 0. 0.71]·5.00m' instead of a bare label."""
    p0 = np.asarray(get_reference_trajectory(0.0, traj)[0], float)
    p1 = np.asarray(get_reference_trajectory(t_end - DT, traj)[0], float)
    d = p1 - p0
    mag = float(np.linalg.norm(d))
    u = d / mag if mag > 1e-9 else d
    return f"move {np.round(u, 2)}·{mag:.2f}m"


def _mark_times(ax, t, xyz, n=TIME_MARKS, color="k", label=True):
    """Drop n evenly-spaced t=Xs markers along a 3-D (x,y,z) path so the spatial plot carries a time axis.
    xyz: (T,>=3) positions on the same time grid as t. Snaps each target time to the nearest sample."""
    if n <= 0 or len(t) < 2:
        return
    t = np.asarray(t)
    for tm in np.linspace(t[0], t[-1], n):
        k = min(int(np.argmin(np.abs(t - tm))), len(xyz) - 1)   # clamp (path may be 1 sample shorter)
        x, y, z = xyz[k, 0], xyz[k, 1], xyz[k, 2]
        ax.plot([x], [y], [z], "o", color=color, ms=4, mfc="white", mew=1.2, zorder=6)
        if label:
            ax.text(x, y, z, f" t={t[k]:.0f}s", fontsize=7, color=color, zorder=7)


def _tracking_panels(t, ref, net_act, opt_act, comp_labels, unit, title):
    """3-panel ref-vs-actual time series (X/Y/Z or roll/pitch/yaw). ref is the shared reference
    (dashed black); net_act/opt_act are actual tracks (either may be None)."""
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for k, lbl in enumerate(comp_labels):
        ax[k].plot(t, ref[:, k], "k--", lw=1.5, label="reference")
        if net_act is not None:
            ax[k].plot(t, net_act[:, k], "C0", lw=1.3, label="net")
        if opt_act is not None:
            ax[k].plot(t, opt_act[:, k], "C3", lw=1.1, ls="--", alpha=0.8, label="optimizer")
        ax[k].set_ylabel(f"{lbl} ({unit})"); ax[k].grid(True)
        if k == 0:
            ax[k].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("Time (s)"); fig.suptitle(title)


def run_episode_opt(env, agent, Bb, L0, traj=None, t_end=None):
    """Closed-loop OPTIMIZER episode on `traj` (mirrors collect_prdot_data's expert rollout).
    Returns the SAME dict shape as deploy_prdot.run_episode so plotting is symmetric.
    `t_end` overrides the loop length (default T_END) — used for the compact SHOWCASE."""
    t_end = T_END if t_end is None else t_end
    obs42, _ = env.reset()
    agent.reset()
    prev_f = np.array([0.0, 0.0, FZ] * N)
    t_hist, load_hist, ref_hist = [], [], []
    rot_hist, rotref_hist = [], []           # load orientation (deg): actual vs reference
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

        ref_t = get_reference_trajectory(t, traj)
        t_hist.append(t)
        load_hist.append(pos.copy())
        ref_hist.append(ref_t[0].copy())
        rot_hist.append(euler_deg(R)); rotref_hist.append(euler_deg(ref_t[2]))
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
                rot=np.array(rot_hist), rot_ref=np.array(rotref_hist),
                dpos=[np.array(p) for p in dpos], dvel=dvel,
                vRi=[np.array(v) for v in vRi_hist],
                lam=[np.array(l) for l in lam_hist],
                A=np.array(A_hist), xi=np.array(xi_hist), solved=np.array(solved_hist),
                track_mean=float(err.mean()), track_max=float(err.max()),
                vmin=float(min(v.min() for v in dvel)))


def _clip_len(a, b):
    """Common leading length of two same-grid time series (guards a 1-sample off-by-one)."""
    return min(len(a), len(b))


def load_track_err(d):
    """Per-step load tracking error magnitude ||load - ref|| (m) and its MSE (m^2)."""
    e = d["load"] - d["ref"]
    mag = np.linalg.norm(e, axis=1)
    return mag, float((e ** 2).sum(1).mean())


def carrier_dev(net, opt):
    """Per-drone deviation of the NET carriers from the OPTIMIZER carriers (the expert is the ref here).
    Returns (per-drone ||net_dpos - opt_dpos|| time series, per-drone MSE m^2, overall MSE)."""
    mags, mses = [], []
    for i in range(N):
        L = _clip_len(net["dpos"][i], opt["dpos"][i])
        e = net["dpos"][i][:L] - opt["dpos"][i][:L]
        mags.append(np.linalg.norm(e, axis=1))
        mses.append(float((e ** 2).sum(1).mean()))
    return mags, mses, float(np.mean(mses))


def lambda_dev(net, opt):
    """Per-drone lambda MSE of net vs optimizer (coordination-signal agreement)."""
    out = []
    for i in range(N):
        L = _clip_len(net["lam"][i], opt["lam"][i])
        out.append(float(((net["lam"][i][:L] - opt["lam"][i][:L]) ** 2).mean()))
    return out


def print_metrics(net, opt):
    """Numbers table: load-tracking MSE (net vs opt), carrier MSE vs optimizer, lambda MSE vs optimizer."""
    _, opt_load_mse = load_track_err(opt)
    print(f"\n{'metric':<30}{'net':>12}{'optimizer':>12}")
    print("-" * 54)
    if net is not None:
        _, net_load_mse = load_track_err(net)
        print(f"{'load track MSE (m^2)':<30}{net_load_mse:>12.5f}{opt_load_mse:>12.5f}")
        print(f"{'load track RMSE (m)':<30}{np.sqrt(net_load_mse):>12.5f}{np.sqrt(opt_load_mse):>12.5f}")
        print(f"{'load track max (m)':<30}{net['track_max']:>12.5f}{opt['track_max']:>12.5f}")
        cmag, cmse, cmse_all = carrier_dev(net, opt)
        print(f"\n{'carrier vs OPTIMIZER (ref)':<30}{'MSE (m^2)':>12}")
        for i in range(N):
            print(f"  {'drone '+str(i+1):<28}{cmse[i]:>12.5f}")
        print(f"  {'ALL drones':<28}{cmse_all:>12.5f}   (RMSE {np.sqrt(cmse_all):.4f} m)")
        lmse = lambda_dev(net, opt)
        print(f"\n{'lambda vs OPTIMIZER (ref)':<30}{'MSE':>12}")
        for i in range(N):
            print(f"  {'drone '+str(i+1):<28}{lmse[i]:>12.5f}")
        print(f"  {'ALL drones':<28}{float(np.mean(lmse)):>12.5f}")
    else:
        print(f"{'load track MSE (m^2)':<30}{'—':>12}{opt_load_mse:>12.5f}")
        print(f"{'load track RMSE (m)':<30}{'—':>12}{np.sqrt(opt_load_mse):>12.5f}")


def plot_compare(net, opt, title=""):
    """Overlay net (solid) vs optimizer (dashed). net may be None (optimizer-only survey)."""
    t = opt["t"]; sfx = f" — {title}" if title else ""
    have_net = net is not None

    # 0a. load POSITION x/y/z — reference vs actual (net + optimizer)
    _tracking_panels(t, opt["ref"], net["load"] if have_net else None, opt["load"],
                     "XYZ", "m", "Load position — ref vs actual" + sfx)

    # 0b. load ORIENTATION roll/pitch/yaw (deg) — reference vs actual
    _tracking_panels(t, opt["rot_ref"], net["rot"] if have_net else None, opt["rot"],
                     ["roll", "pitch", "yaw"], "deg", "Load orientation — ref vs actual" + sfx)

    # 0c. LOAD TRACKING ERROR magnitude ||load - ref|| over time (net vs optimizer) + mean lines.
    plt.figure(figsize=(11, 4.5))
    om, omse = load_track_err(opt)
    plt.plot(t, om, "C3", lw=1.2, ls="--", alpha=0.85, label=f"optimizer (MSE {omse:.4f})")
    plt.axhline(om.mean(), c="C3", ls=":", lw=1.0, alpha=0.6)
    if have_net:
        nm, nmse = load_track_err(net)
        plt.plot(t[:len(nm)], nm, "C0", lw=1.4, label=f"net (MSE {nmse:.4f})")
        plt.axhline(nm.mean(), c="C0", ls=":", lw=1.0, alpha=0.6)
    plt.xlabel("Time (s)"); plt.ylabel("‖load − ref‖ (m)")
    plt.title("Load tracking error over time" + sfx)
    plt.legend(fontsize=9); plt.grid(True)

    # 0d. CARRIER deviation from the OPTIMIZER (expert) per drone over time — where the net's drones drift.
    if have_net:
        cmag, cmse, cmse_all = carrier_dev(net, opt)
        plt.figure(figsize=(11, 4.5))
        for i in range(N):
            plt.plot(t[:len(cmag[i])], cmag[i], color=f"C{i}", lw=1.3,
                     label=f"drone {i+1} (MSE {cmse[i]:.4f})")
        plt.xlabel("Time (s)"); plt.ylabel("‖net − opt‖ carrier pos (m)")
        plt.title(f"Carrier deviation from optimizer  (all-drone MSE {cmse_all:.4f} m²)" + sfx)
        plt.legend(fontsize=8); plt.grid(True)

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

    # 3. drone XYZ trajectories + load (3-D), with t=Xs numbers along each drone path
    s0 = 2                                                        # skip the t=0 sample (weird init jump)
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(N):
        dp = (net if have_net else opt)["dpos"][i]               # path the time-numbers ride on
        if have_net:
            ax.plot(net["dpos"][i][s0:, 0], net["dpos"][i][s0:, 1], net["dpos"][i][s0:, 2],
                    color=f"C{i}", lw=1.4, label=f"drone {i+1} net")
        ax.plot(opt["dpos"][i][s0:, 0], opt["dpos"][i][s0:, 1], opt["dpos"][i][s0:, 2],
                color=f"C{i}", lw=1.2, ls="--", alpha=0.7, label=f"drone {i+1} opt")
        _mark_times(ax, t[s0:], dp[s0:], color=f"C{i}", label=True)   # NUMBERS on the drone trajectories
    if have_net:
        ax.plot(net["load"][s0:, 0], net["load"][s0:, 1], net["load"][s0:, 2], "k", lw=2, label="load net")
    ax.plot(opt["load"][s0:, 0], opt["load"][s0:, 1], opt["load"][s0:, 2],
            color="gray", lw=2, ls="--", label="load opt")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Drone XYZ trajectories — net (solid) vs optimizer (dashed)" + sfx)
    ax.legend(ncol=2, fontsize=8)

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
    if USE_MOVE:
        d = np.asarray(MOVE_DIR, float)
        d = d / (np.linalg.norm(d) + 1e-12)
        pos_delta = MOVE_MAG * d
        traj = make_quintic_pose(pos_delta, np.zeros(3), ramp=MOVE_RAMP,
                                 hold=HOLD, base_pos=np.asarray(BASE_POS, float))
        t_end = HOLD + MOVE_RAMP + MOVE_MARGIN
        title = f"move {np.round(MOVE_DIR, 2)}·{MOVE_MAG:g}m  ramp {MOVE_RAMP:g}s"
        print(f"QUINTIC MOVE: dir {np.round(d, 3)} · {MOVE_MAG:g} m = dpos {np.round(pos_delta, 2)}  "
              f"ramp {MOVE_RAMP:g}s  t_end {t_end:.0f}s\n")
    elif SHOWCASE is not None:
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

    if not USE_MOVE:                       # title shows the actual MOVE (dir·mag), not the bare label
        title = f"{move_descr(traj, t_end)}  (t_end {t_end:.0f}s)"

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

    print_metrics(net, opt)

    plot_compare(net, opt, title=title)
    plt.show()


if __name__ == "__main__":
    main()
