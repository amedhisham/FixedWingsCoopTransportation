"""
overlay_classical_f2.py — PPT figure: pure classical controller vs F2 decentralized
(zero-noise sanity) on the SAME axes.

Two rollouts from the same reset/reference:
  A) CLASSICAL  — one central ClassicalAgent (optimizer + wrench) drives the plant
                  (the val_agent.run_agent pipeline).
  B) F2 SANITY  — ResidualMARLEnv at zero noise / zero residual: four decentralized
                  LocalModelAgents running the frozen lambda-net (the sanity_f2 setup).

Overlays, styled for slides (classical = solid, F2 = dashed, same colour per drone):
  1. drone XY trajectories (+ load path)
  2. load tracking X/Y/Z vs reference
Saved to PNG (overlay_xy.png, overlay_load.png) and shown.
"""

import numpy as np
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from controller import error_calculation, get_reference_trajectory
from residual_marl_env import ResidualMARLEnv

N = 4
DT, T_END = 0.01, 25.0
EPS = 0.25
PHASES = np.array([0.0, np.pi / 2, 0.0, np.pi / 2])
LLC_ALPHA = DT / (0.2 + DT)
FZ = 0.7 * 9.81 / 4
DRONE_COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]   # one per drone


def read_params(env):
    J = np.array(env.fmu.getReal([env.vrs[f"Load_Inertia_Matrix[1,{i}]"] for i in range(1, 10)])).reshape((3, 3), order="F")
    Bb = np.array(env.fmu.getReal([env.vrs[f"Attachment_Point_Vectors[1,{i}]"] for i in range(1, 13)])).reshape((N, 3))
    m = env.fmu.getReal([env.vrs["Load_Mass"]])[0]
    L0 = env.fmu.getReal([env.vrs["Cable_Resting_Length"]])[0]
    return J, Bb, m, L0


def run_classical():
    """Pure central classical controller driving the plant (val_agent.run_agent + logging)."""
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    prev_f = np.array([0.0, 0.0, FZ] * N)
    t_hist, load, ref = [], [], []
    dpos = [[] for _ in range(N)]
    t = 0.0
    while t < T_END - 1e-9:
        pos = obs[0:3]
        R = np.round(obs[3:12].reshape((3, 3), order="C"), 6)
        vel, w = obs[12:15], obs[15:18]
        f, _, _ = agent.compute_forces(pos, vel, R, w, t)
        ff = LLC_ALPHA * f + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        t_hist.append(t); load.append(pos.copy()); ref.append(get_reference_trajectory(t)[0].copy())
        for i in range(N):
            dpos[i].append(obs[18 + 3 * i: 18 + 3 * i + 3].copy())
        obs, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()
    return (np.array(t_hist), np.array(load), np.array(ref),
            [np.array(p) for p in dpos])


def run_f2():
    """F2 decentralized at zero noise / zero residual (sanity_f2 setup)."""
    env = ResidualMARLEnv()                      # defaults: zero noise, analytic lambda-net
    env.reset(seed=0)
    zero = {a: np.zeros(3, dtype=np.float32) for a in env.possible_agents}
    t_hist, load = [], []
    dpos = [[] for _ in range(N)]
    while env.agents:
        env.step(zero)
        s = env.state()
        t_hist.append(env.t); load.append(s[0:3].copy())
        for i in range(N):
            dpos[i].append(s[18 + 3 * i: 18 + 3 * i + 3].copy())
    env.close()
    return np.array(t_hist), np.array(load), [np.array(p) for p in dpos]


def main():
    tc, load_c, ref, dpos_c = run_classical()
    tf, load_f, dpos_f = run_f2()

    # ---- Fig 1: drone XY trajectories (classical thick+faded UNDER, F2 thin dashed ON TOP
    #      so where the net SMOOTHS the classical's sharp turn you see both) ----
    def draw_xy(ax):
        for i in range(N):
            ax.plot(dpos_c[i][:, 0], dpos_c[i][:, 1], "-", color=DRONE_COLORS[i], lw=3.4,
                    alpha=0.35, solid_capstyle="round", label=f"Drone {i+1}")
            ax.plot(dpos_f[i][:, 0], dpos_f[i][:, 1], "--", color=DRONE_COLORS[i], lw=1.4)
        ax.plot(load_c[:, 0], load_c[:, 1], "-", color="k", lw=2.6, alpha=0.35, label="Load")
        ax.plot(load_f[:, 0], load_f[:, 1], "--", color="k", lw=1.4)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.axis("equal"); ax.grid(True)
        style = [plt.Line2D([], [], color="k", ls="-", lw=3.4, alpha=0.35, label="Classical (central)"),
                 plt.Line2D([], [], color="k", ls="--", lw=1.4, label="F2 decentralized")]
        leg = ax.legend(handles=style, loc="upper left", frameon=True, fontsize=9)
        ax.add_artist(leg)
        ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig1, ax = plt.subplots(figsize=(9, 7))
    draw_xy(ax)
    ax.set_title("Drone XY trajectories — classical vs F2 decentralized (zero noise)")
    plt.tight_layout(); plt.savefig("overlay_xy.png", dpi=150)

    # zoomed version — first couple of loiter loops, where the sharp/smooth turn shows
    fig1z, axz = plt.subplots(figsize=(9, 6))
    draw_xy(axz)
    axz.set_xlim(-1.6, 4.2); axz.set_ylim(-1.6, 1.6)
    axz.set_title("Drone XY — first loops (classical sharp cusp vs F2 smoothed)")
    plt.tight_layout(); plt.savefig("overlay_xy_zoom.png", dpi=150)

    # ---- Fig 2: load tracking X/Y/Z ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(tc, ref[:, k], ":", color="0.5", lw=1.5, label="reference")
        ax.plot(tc, load_c[:, k], "-", color="#1f77b4", lw=1.8, label="classical (central)")
        ax.plot(tf, load_f[:, k], "--", color="#d95f02", lw=1.8, label="F2 decentralized")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True)
        if k == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=9)
    axes[2].set_xlabel("Time (s)")
    fig.suptitle("Load tracking — classical vs F2 decentralized (zero noise)")
    plt.tight_layout(); plt.savefig("overlay_load.png", dpi=150)

    # numeric context
    ec = np.linalg.norm(load_c - ref, axis=1)
    n = min(len(load_f), len(ref))
    ef = np.linalg.norm(load_f[:n] - ref[:n], axis=1)
    print(f"classical: mean track {ec.mean():.4f} m  max {ec.max():.4f} m")
    print(f"F2 sanity: mean track {ef.mean():.4f} m  max {ef.max():.4f} m")
    print("saved overlay_xy.png, overlay_load.png")
    plt.show()


if __name__ == "__main__":
    main()
