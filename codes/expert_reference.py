"""
expert_reference.py — precompute the CLEAN expert's per-drone loiter path.

Runs the central ClassicalAgent once on the (noise-free) plant and stores every drone's
position/velocity over the trajectory. ResidualMARLEnv loads this as the reference the
RL reward tracks: r_i penalizes distance to the NEAREST point on this path (phase-free
"stay on the loiter loop"), so it must be regenerated whenever the reference trajectory
(get_reference_trajectory) changes. Single trajectory -> run once, save expert_ref.npz.
"""

import numpy as np
from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from controller import get_reference_trajectory
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ

OUT = "expert_ref.npz"


def main():
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    prev_f = np.array([0.0, 0.0, FZ] * N)
    dpos = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    load = []
    t = 0.0
    while t < T_END - 1e-9:
        pos = obs[0:3]
        R = np.round(obs[3:12].reshape((3, 3), order="C"), 6)
        vel, w = obs[12:15], obs[15:18]
        f, _, _ = agent.compute_forces(pos, vel, R, w, t)
        ff = LLC_ALPHA * f + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        for i in range(N):
            dpos[i].append(obs[18 + 3 * i: 18 + 3 * i + 3].copy())
            dvel[i].append(obs[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3].copy())
        load.append(pos.copy())
        obs, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    dpos = np.array(dpos)   # (N, T, 3)
    dvel = np.array(dvel)   # (N, T, 3)
    load = np.array(load)   # (T, 3)
    np.savez(OUT, dpos=dpos, dvel=dvel, load=load)
    print(f"saved {OUT}  dpos {dpos.shape}  (per-drone loiter path over {dpos.shape[1]} steps)")


if __name__ == "__main__":
    main()
