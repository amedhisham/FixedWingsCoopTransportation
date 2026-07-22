"""
desync_baseline.py — quantify the F2 job across several desync configs (zero residual).

Same structure as demo_desync.py (a run_episode helper + interactive plots), but instead
of ONE disturbance it sweeps a few CONFIGS and prints the load-tracking degradation table
— the number the learned residual has to beat. demo_desync is the single-config visual
deep-dive; this is the multi-config scoreboard.

There is no learning here: delta_f = 0, and we just turn on the desync knobs to confirm
the disturbance actually degrades load tracking (so the residual RL has a real job).
"""

import numpy as np
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv
from controller import get_reference_trajectory

N = 4
EPSILON = 0.25

# label -> desync kwargs (empty == coherent baseline). Noise is per-drone + AR(1)-
# correlated, so the four load views diverge -> the G_i/N_i/w_d_i disagree -> internal
# forces stop cancelling -> load disturbed. clock_offset mostly sits in the nullspace
# (benign); pos/vel noise + ctrl_delay are the load-disturbing ones.
CONFIGS = {
    "zero (baseline)":        {},
    "obs noise (pos+vel)":    dict(pos_noise=0.03, vel_noise=0.10, noise_corr=0.995),
    "ctrl delay (staggered)": dict(ctrl_delay=[0, 2, 2, 1]),
    "combined":               dict(pos_noise=0.03, vel_noise=0.10, noise_corr=0.995,
                                   ctrl_delay=[0, 2, 2, 1]),
}


def run_episode(**kwargs):
    """Roll out one episode with zero residual actions; collect histories. Same shape as
    demo_desync.run_episode, plus the tracking-error series for the scoreboard."""
    env = ResidualMARLEnv(n_carriers=N, epsilon=EPSILON, **kwargs)
    env.reset()
    zero = {a: np.zeros(3, dtype=np.float32) for a in env.possible_agents}

    t_hist, load_hist, ref_hist = [], [], []
    dpos = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]

    t = 0.0
    while True:
        s = env.state()                       # global 42-D state at current t
        t_hist.append(t)
        load_hist.append(s[0:3].copy())
        ref_hist.append(get_reference_trajectory(t)[0].copy())
        for i in range(N):
            dpos[i].append(s[18 + 3 * i: 18 + 3 * i + 3].copy())
            dvel[i].append(np.linalg.norm(s[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        env.step(zero)
        t += env.dt
        if not env.agents:                    # truncated -> episode over
            break
    env.close()

    load = np.array(load_hist); ref = np.array(ref_hist)
    return dict(
        t=np.array(t_hist), load=load, ref=ref,
        dpos=[np.array(p) for p in dpos], dvel=[np.array(v) for v in dvel],
        err=np.linalg.norm(load - ref, axis=1),
    )


if __name__ == "__main__":
    print("F2 desync baseline (zero residual) — load tracking degradation:")
    results = {}
    for label, kw in CONFIGS.items():
        r = run_episode(**kw)
        results[label] = r
        print(f"  {label:24s}  mean {r['err'].mean():.4f}   max {r['err'].max():.4f} m")

    # Worst non-baseline config drives the detail plots; overlay the coherent run.
    worst = max((l for l in results if l != "zero (baseline)"),
                key=lambda l: results[l]["err"].mean())
    des, coh = results[worst], results["zero (baseline)"]

    # 1. Load tracking-error norm over time — all configs overlaid.
    plt.figure()
    for label, r in results.items():
        plt.plot(r["t"], r["err"], label=label)
    plt.xlabel("Time (s)"); plt.ylabel("||load - reference|| (m)")
    plt.title("Load tracking error vs desync (zero residual)"); plt.legend(); plt.grid(True)

    # 2. Load position vs reference, worst desync vs coherent.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(des["t"], des["ref"][:, k], "k--", lw=2, label="reference")
        ax.plot(des["t"], des["load"][:, k], "r", label=f"desynced ({worst})")
        ax.plot(coh["t"], coh["load"][:, k], "g", alpha=0.7, label="coherent")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)"); fig.suptitle("Load position tracking — worst desync vs coherent")

    # 3. Drone velocity norms (worst desync).
    plt.figure()
    for i in range(N):
        plt.plot(des["t"], des["dvel"][i], label=f"Drone {i+1}")
    plt.axhline(EPSILON, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms — {worst}"); plt.legend(); plt.grid(True)

    # 4. Drone XY trajectories (worst desync).
    plt.figure(figsize=(8, 6))
    for i in range(N):
        plt.plot(des["dpos"][i][:, 0], des["dpos"][i][:, 1], label=f"Drone {i+1}")
    plt.plot(des["load"][:, 0], des["load"][:, 1], "k--", lw=2, label="Load")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title(f"Drone XY trajectories — {worst}"); plt.legend(); plt.grid(True); plt.axis("equal")

    plt.show()
