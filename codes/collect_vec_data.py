"""
collect_vec_data.py — data for the WHOLE-VECTOR policy (optimizer-style I/O).

The optimizer maps (load state, phases, clock) -> the whole lambda vector, never
observing carriers. We record exactly that mapping so we can train a network that
imitates it: (load 18 + clock 14) -> lambda[n]. We ALSO store each drone's own
state, so train_vec.py can fit the with-own-state variant and compare R^2.

Per step it saves:
    load  (18)      : [pos, R(row-major), lin_vel, ang_vel]  (R rounded, as in F1)
    clock (14)      : the Fourier clock bank
    drone (n, 6)    : each carrier's own [pos, vel]  (for the comparison only)
    lam   (n,)      : the optimizer's full lambda vector  (the target)
"""

import numpy as np
from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from collect_il_data import clock_features, read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ

BYPASS_OPT = False   # adaptive optimizer (the real sweeping target)
OUT = "vec_dataset.npz"


def collect():
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    prev_f = np.array([0.0, 0.0, FZ] * N)
    load_rows, clock_rows, drone_rows, lam_rows = [], [], [], []

    t = 0.0
    while t < T_END - 1e-9:
        load18 = obs42[0:18].copy()
        load18[3:12] = np.round(load18[3:12], 6)
        clk = clock_features(t)

        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]
        f_full, _, lam = agent.compute_forces(pos, vel, R, angvel, t, bypass_opt=BYPASS_OPT)

        dstates = np.zeros((N, 6))
        for i in range(N):
            dstates[i, :3] = obs42[18 + 3 * i: 18 + 3 * i + 3]
            dstates[i, 3:] = obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]

        load_rows.append(load18)
        clock_rows.append(clk)
        drone_rows.append(dstates)
        lam_rows.append(lam)

        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    return (np.asarray(load_rows, dtype=np.float32),
            np.asarray(clock_rows, dtype=np.float32),
            np.asarray(drone_rows, dtype=np.float32),
            np.asarray(lam_rows, dtype=np.float32))


if __name__ == "__main__":
    load, clock, drone, lam = collect()
    np.savez(OUT, load=load, clock=clock, drone=drone, lam=lam)
    print(f"saved {OUT}   steps={len(load)}")
    print(f"  load {load.shape}  clock {clock.shape}  drone {drone.shape}  lam {lam.shape}")
    print(f"  lambda range [{lam.min():.3f}, {lam.max():.3f}]  std {lam.std():.3f}")
