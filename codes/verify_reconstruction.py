"""
verify_reconstruction.py — check the kinematic carrier-state reconstruction.

For the zero-communication F2 setup, each drone must RECONSTRUCT every carrier's
position/velocity from the LOAD state + geometry + the (filtered) forces — the
paper's p_R / p_R_dot relations — instead of observing them. This script verifies
those relations reproduce the plant's ACTUAL drone states.

Per drone i, from the FILTERED force f (what actually enters the FMU) and its
derivative fdot:
    q_i   = f_i / ||f_i||                    # cable direction (force is along the cable)
    p_R_i = p_L + R @ Bb_i + L0 * q_i        # attachment point + cable
    v_Li  = v_L + R @ (omega x Bb_i)         # attachment-point velocity
    Pi_i  = I - q_i q_i^T
    v_R_i = v_Li + (L0/||f_i||) * Pi_i @ fdot_i   # = optimizer.py's v_Ri (e+g == fdot)

Reconstruction is KINEMATIC (force -> carrier state), so it's independent of what
computes the force; we drive with the fast bypass controller. If p_R/v_R overlay
the plant's Drone_Positions / Drones_LinVelocity, the F2 local model is exact and
each drone can rebuild the whole system from its own load view — zero comms.
"""

import numpy as np
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from collect_il_data import read_params

N = 4
DT, T_END = 0.01, 25.0
EPS = 0.25
PHASES = np.array([0.0, np.pi / 2, 0.0, np.pi / 2])
LLC_ALPHA = DT / (0.2 + DT)
FZ = 0.7 * 9.81 / 4
BYPASS_OPT = True   # fast; reconstruction is controller-independent (pure kinematics)


def reconstruct(pL, R, vL, omega, ff, deriv, Bb, L0):
    """Carrier positions (N,3) and velocities (N,3) from load state + filtered force."""
    pR = np.zeros((N, 3))
    vR = np.zeros((N, 3))
    for i in range(N):
        f_i = ff[3 * i: 3 * i + 3]
        fd_i = deriv[3 * i: 3 * i + 3]
        T = np.linalg.norm(f_i)
        q = f_i / T
        pR[i] = pL + R @ Bb[i] + L0 * q
        vLi = vL + R @ np.cross(omega, Bb[i])
        Pi = np.eye(3) - np.outer(q, q)
        vR[i] = vLi + (L0 / T) * Pi @ fd_i
    return pR, vR


def main():
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    prev_f = np.array([0.0, 0.0, FZ] * N)
    t = 0.0
    t_hist = []
    perr = [[] for _ in range(N)]      # position error norm
    verr = [[] for _ in range(N)]      # velocity error norm
    recp = [[] for _ in range(N)]; actp = [[] for _ in range(N)]   # for overlay (drone 0)

    while t < T_END - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        f_full, _, _ = agent.compute_forces(pos, vel, R, angvel, t, bypass_opt=BYPASS_OPT)
        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()

        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT

        # After the step: the new obs42 drone states are the response to ff.
        npos = obs42[0:3]
        nR = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        nvel, nangvel = obs42[12:15], obs42[15:18]
        pR, vR = reconstruct(npos, nR, nvel, nangvel, ff, deriv, Bb, L0)

        act_pos = obs42[18:18 + 3 * N].reshape(N, 3)
        act_vel = obs42[18 + 3 * N:18 + 6 * N].reshape(N, 3)

        t_hist.append(t)
        for i in range(N):
            perr[i].append(np.linalg.norm(pR[i] - act_pos[i]))
            verr[i].append(np.linalg.norm(vR[i] - act_vel[i]))
            recp[i].append(pR[i].copy()); actp[i].append(act_pos[i].copy())
    env.close()

    t_hist = np.array(t_hist)
    perr = [np.array(e) for e in perr]; verr = [np.array(e) for e in verr]

    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    warm = int(0.5 / DT)   # skip 0.5 s startup transient
    print("reconstruction error (||recon - plant||):")
    for i in range(N):
        p, v = perr[i], verr[i]
        print(f"  drone {i+1}:  pos RMSE {rms(p):.6f}  steady {rms(p[warm:]):.6f}  "
              f"max {p.max():.4f} @ t={t_hist[p.argmax()]:.2f}s"
              f"  |  vel RMSE {rms(v):.6f}  steady {rms(v[warm:]):.6f}  "
              f"max {v.max():.4f} @ t={t_hist[v.argmax()]:.2f}s")

    # Fig 1: position error over time.
    plt.figure()
    for i in range(N):
        plt.plot(t_hist, perr[i], label=f"Drone {i+1}")
    plt.xlabel("Time (s)"); plt.ylabel("||p_R recon - actual|| (m)")
    plt.title("Carrier position reconstruction error"); plt.legend(); plt.grid(True)

    # Fig 2: velocity error over time.
    plt.figure()
    for i in range(N):
        plt.plot(t_hist, verr[i], label=f"Drone {i+1}")
    plt.xlabel("Time (s)"); plt.ylabel("||v_R recon - actual|| (m/s)")
    plt.title("Carrier velocity reconstruction error"); plt.legend(); plt.grid(True)

    # Fig 3: overlay for drone 1 (recon vs actual position, XYZ).
    recp0 = np.array(recp[0]); actp0 = np.array(actp[0])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(t_hist, actp0[:, k], "b", label="plant")
        ax.plot(t_hist, recp0[:, k], "r--", label="reconstructed")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)"); fig.suptitle("Drone 1 position — plant vs reconstruction")

    plt.show()


if __name__ == "__main__":
    main()
