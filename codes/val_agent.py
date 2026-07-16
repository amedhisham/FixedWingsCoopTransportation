"""
val_agent.py — verify ClassicalAgent == inline classical pipeline.

Runs two independent FMU rollouts from the same reset:
  A) the inline functions (as in main_env.py, using module-singleton state)
  B) a single ClassicalAgent (instance state)
and reports the largest force / load-position discrepancy between them.
Faithful refactor  ==>  differences at floating-point noise level (~1e-12).
"""

import numpy as np

from fmu_plant_env import FMUPlantEnv
from controller import error_calculation, wrench_controller
from optimizer import cable_force_calculation, init_optimizer, optimizer
from classical_agent import ClassicalAgent

# --- shared config ---
N = 4
EPS = 0.25
PHASES = np.array([0, np.pi / 2, 0, np.pi / 2])
DT, T_END = 0.01, 25.0
LLC_ALPHA = DT / (0.2 + DT)


def read_params(env):
    J = np.array(env.fmu.getReal([env.vrs[f"Load_Inertia_Matrix[1,{i}]"] for i in range(1, 10)])).reshape((3, 3), order="F")
    Bb = np.array(env.fmu.getReal([env.vrs[f"Attachment_Point_Vectors[1,{i}]"] for i in range(1, 13)])).reshape((N, 3))
    m = env.fmu.getReal([env.vrs["Load_Mass"]])[0]
    L0 = env.fmu.getReal([env.vrs["Cable_Resting_Length"]])[0]
    return J, Bb, m, L0


def unpack(obs):
    R = np.round(obs[3:12].reshape((3, 3), order="C"), 6)
    return obs[0:3], R, obs[12:15], obs[15:18]


def run_inline():
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    solver = init_optimizer(L0, N, 0, 0, PHASES)
    prev_f = np.array([0.0, 0.0, 0.7 * 9.81 / 4] * N)
    forces, loads, t = [], [], 0.0
    while t < T_END - 1e-9:
        pos, R, vel, w = unpack(obs)
        ep, eR, ev, ew = error_calculation(pos, vel, R, w, t)
        w_d = wrench_controller(ep, eR, ev, ew, w, J, m, Bb, DT, N, 0, None)
        lam, _ = optimizer(solver, t, R, vel, w, w_d, Bb, EPS, DT, N, PHASES, 0)
        f, _ = cable_force_calculation(R, Bb, w_d, lam, N)
        ff = LLC_ALPHA * f + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs, *_ = env.step(np.concatenate([ff, deriv]))
        forces.append(ff.copy()); loads.append(pos.copy()); t += DT
    env.close()
    return np.array(forces), np.array(loads)


def run_agent():
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)
    prev_f = np.array([0.0, 0.0, 0.7 * 9.81 / 4] * N)
    forces, loads, t = [], [], 0.0
    while t < T_END - 1e-9:
        pos, R, vel, w = unpack(obs)
        f, _ = agent.compute_forces(pos, vel, R, w, t)
        ff = LLC_ALPHA * f + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs, *_ = env.step(np.concatenate([ff, deriv]))
        forces.append(ff.copy()); loads.append(pos.copy()); t += DT
    env.close()
    return np.array(forces), np.array(loads)


if __name__ == "__main__":
    fa, la = run_inline()
    fb, lb = run_agent()
    print(f"max |force_inline - force_agent| = {np.abs(fa - fb).max():.3e}")
    print(f"max |load_inline  - load_agent | = {np.abs(la - lb).max():.3e}")
