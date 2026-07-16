"""
demo_desync.py — run the F2 wrapper with ZERO residual actions and SEE what
decentralization desync does, with interactive main.py-style plots.

There is no learning here. We drive the 4-expert stitch with delta_f = 0 and turn
on the desync knobs, to check the disturbance actually degrades load tracking
(i.e. that the residual RL will have a real job to do).

Run:   python demo_desync.py
Edit the DESYNC dict below to change the disturbance. Set COMPARE=False to skip
the coherent baseline overlay (twice as fast).
"""

import numpy as np
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv
from controller import get_reference_trajectory

N = 4
EPSILON = 0.25
DESYNC_ON = True   # set False to see ONLY the coherent (no-noise, no-delay) baseline
COMPARE = False     # when desync is on, overlay a coherent run so degradation is obvious

# Every disturbance knob is exposed here (all-zero == coherent baseline).
# Note: with ZERO residual actions, own_noise has no visible effect (no policy is
# reading the observation yet); load_noise / actuation_noise / delay / clock_offset do.
DESYNC = dict(
    ctrl_delay=[0, 2, 2, 1],                 # steps of load-estimate delay (x10 ms) per drone
    clock_offset=[0.0, 0.00, -0.00, 0.00],   # seconds added to each drone's clock
    pos_noise=0.03,                          # load position sensing noise, m   (shared expert + obs)
    rot_noise=0.01,                          # load orientation sensing noise
    vel_noise=0.10,                          # load linear-velocity sensing noise, m/s
    angvel_noise=0.05,                       # load angular-velocity sensing noise, rad/s
    noise_corr=0.995,                         # temporal smoothness of sensing noise (0=white, ->1=smooth)
    own_noise=0.0,                           # drone's own pos/vel sensing noise (obs only)
    actuation_noise=0.0,                     # noise on the commanded forces
)


def run_episode(**kwargs):
    """Roll out one episode with zero residual actions; collect histories."""
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
    return dict(
        t=np.array(t_hist), load=np.array(load_hist), ref=np.array(ref_hist),
        dpos=[np.array(p) for p in dpos], dvel=[np.array(v) for v in dvel],
    )


if __name__ == "__main__":
    if DESYNC_ON:
        des = run_episode(**DESYNC)
        coh = run_episode() if COMPARE else None
    else:
        des = run_episode()            # coherent baseline only
        coh = None

    # 1. Drone velocity norms (desynced).
    plt.figure()
    for i in range(N):
        plt.plot(des["t"], des["dvel"][i], label=f"Drone {i+1}")
    plt.axhline(EPSILON, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title("Drone velocity norms (desynced)"); plt.legend(); plt.grid(True)

    # 2. Load position vs reference, per axis (coherent vs desynced).
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(des["t"], des["ref"][:, k], "k--", lw=2, label="reference")
        ax.plot(des["t"], des["load"][:, k], "r", label="desynced" if DESYNC_ON else "load")
        if coh is not None:
            ax.plot(coh["t"], coh["load"][:, k], "g", alpha=0.7, label="coherent")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)"); fig.suptitle("Load position tracking")

    # 3. Load tracking-error norm over time (coherent vs desynced).
    plt.figure()
    plt.plot(des["t"], np.linalg.norm(des["load"] - des["ref"], axis=1), "r", label="desynced" if DESYNC_ON else "load")
    if coh is not None:
        plt.plot(coh["t"], np.linalg.norm(coh["load"] - coh["ref"], axis=1), "g", label="coherent")
    plt.xlabel("Time (s)"); plt.ylabel("||load - reference|| (m)")
    plt.title("Load tracking error"); plt.legend(); plt.grid(True)

    # 4. Drone XY trajectories (desynced).
    plt.figure(figsize=(8, 6))
    for i in range(N):
        plt.plot(des["dpos"][i][:, 0], des["dpos"][i][:, 1], label=f"Drone {i+1}")
    plt.plot(des["load"][:, 0], des["load"][:, 1], "k--", lw=2, label="Load")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("Drone XY trajectories (desynced)"); plt.legend(); plt.grid(True); plt.axis("equal")

    plt.show()
