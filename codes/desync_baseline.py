"""
desync_baseline.py — quantify the F2 job (zero residual, various desync).

At zero noise the four local replicas are coherent -> load tracks like deploy_prdot
(~0.08 m). The whole reason F2 exists: under OBSERVATION DESYNC each drone builds its
own (divergent) load view -> its G_i / N_i / w_d_i disagree -> the internal forces stop
cancelling -> the load is disturbed. This script turns on the desync knobs with delta_f
= 0 and measures how far tracking degrades from baseline. That degradation is exactly
what the learned residual has to claw back — the number to beat in step 2.

Edit CONFIGS to taste. Runs each, prints the tracking table, then plots the tracking
error over time (all configs) + the worst case's load + drone velocities.
"""

import numpy as np
import matplotlib.pyplot as plt
from residual_marl_env import ResidualMARLEnv
from controller import get_reference_trajectory

N = 4

# Desync configs (all at zero residual). Noise is per-drone + AR(1)-correlated, so the
# four load views diverge -> forces stop cancelling. clock_offset mostly sits in the
# nullspace (benign); pos/vel noise + ctrl_delay are the real load-disturbing ones.
CONFIGS = {
    "zero (baseline)":        {},
    "obs noise (pos+vel)":    dict(pos_noise=0.03, vel_noise=0.08, noise_corr=0.95),
    "ctrl delay (staggered)": dict(ctrl_delay=[0, 2, 4, 6]),
    "combined":               dict(pos_noise=0.03, vel_noise=0.08, noise_corr=0.95,
                                   ctrl_delay=[0, 2, 4, 6]),
}


def run(label, **kw):
    env = ResidualMARLEnv(**kw)
    obs, _ = env.reset(seed=0)
    zero = {a: np.zeros(3, dtype=np.float32) for a in env.possible_agents}

    t_hist, load_hist, ref_hist = [], [], []
    dvel = [[] for _ in range(N)]
    blew = False
    while env.agents:
        obs, r, term, trunc, infos = env.step(zero)
        s = env.state()
        if np.isnan(s).any() or np.abs(s).max() > 1e4:
            blew = True
            break
        t_hist.append(env.t)
        load_hist.append(s[0:3].copy())
        ref_hist.append(get_reference_trajectory(env.t)[0].copy())
        for i in range(N):
            dvel[i].append(np.linalg.norm(s[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
    env.close()

    load = np.array(load_hist); ref = np.array(ref_hist)
    err = np.linalg.norm(load - ref, axis=1) if len(load) else np.array([np.nan])
    tag = "  BLEW UP" if blew else ""
    print(f"  {label:24s}  mean {err.mean():.4f}   max {err.max():.4f} m{tag}")
    return dict(t=np.array(t_hist), load=load, ref=ref, err=err,
                dvel=[np.array(v) for v in dvel], label=label, blew=blew)


def main():
    print("F2 desync baseline (zero residual) — load tracking degradation:")
    results = [run(label, **kw) for label, kw in CONFIGS.items()]

    # Worst non-baseline case (highest mean error, ignoring blowups if any survived).
    worst = max(results[1:], key=lambda r: np.nanmean(r["err"]))

    # 1. Tracking error over time, all configs overlaid.
    plt.figure()
    for r in results:
        if len(r["t"]):
            plt.plot(r["t"], r["err"], label=r["label"])
    plt.xlabel("Time (s)"); plt.ylabel("||load - ref|| (m)")
    plt.title("Load tracking error vs desync (zero residual)"); plt.legend(); plt.grid(True)

    # 2. Worst case: load XYZ vs reference.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(worst["t"], worst["ref"][:, k], "k--", lw=2, label="reference")
        ax.plot(worst["t"], worst["load"][:, k], "r", label="load (desync)")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"Load tracking under desync — {worst['label']}")

    # 3. Worst case: drone velocity norms.
    plt.figure()
    for i in range(N):
        plt.plot(worst["t"], worst["dvel"][i], label=f"Drone {i+1}")
    plt.axhline(0.25, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms under desync — {worst['label']}")
    plt.legend(); plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
