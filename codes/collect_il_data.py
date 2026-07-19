"""
collect_il_data.py — build the imitation dataset for Formulation 1 (lean version).

Under COHERENCE the four experts are identical, so we run ONE ClassicalAgent
(one CasADi solve/step instead of four) and build the four per-drone observations
from the single coherent state. Per drone per step we record:
    (local observation o_i in R^24,  optimizer coefficient lambda_i in R)

Faithfulness to the validated wrapper is checked by val_collector.py.

Caveats (known, to be addressed later):
  - single fixed reference trajectory -> narrow dataset (needs randomization).
  - whether lambda is predictable from o_i alone is what the IL fit will reveal.
"""

import numpy as np
from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent

N = 4
DT, T_END = 0.01, 25.0
EPS = 0.25
PHASES = np.array([0.0, np.pi / 2, 0.0, np.pi / 2])
LLC_ALPHA = DT / (0.2 + DT)
FZ = 0.7 * 9.81 / 4
OUT = "il_dataset.npz"

# Bypass the optimizer: lambda becomes a FIXED sinusoid (constant A, xi). This is
# the clean "easy part" phase test and skips CasADi entirely (fast). Set False to
# collect the adaptive optimizer's lambda (the harder, velocity-dependent target).
# The clock bank exists to handle the ADAPTIVE (sweeping) frequency, so warm-start
# on the real target -> False.
BYPASS_OPT = False

# Common phase clock (Fourier bank). Fixed frequencies bracketing the optimizer's
# xi (~2), so the shared net can linearly combine them to match the slow sweep.
# Same clock for all four drones (coherence); desync (F2) later perturbs each t.
CLOCK_OMEGAS = np.arange(1.5, 3.0 + 1e-9, 0.25)   # 7 freqs -> 14 clock dims


def clock_features(t):
    """[sin(w t), cos(w t)] over the bank. sin+cos both are needed: the pair is a
    unique point on the unit circle, so phase is unambiguous (sin alone folds
    theta and pi-theta together and reintroduces label collisions)."""
    ph = CLOCK_OMEGAS * t
    return np.concatenate([np.sin(ph), np.cos(ph)]).astype(np.float32)


def read_params(env):
    vrs, fmu = env.vrs, env.fmu
    J = np.array(fmu.getReal([vrs[f"Load_Inertia_Matrix[1,{i}]"] for i in range(1, 10)])).reshape((3, 3), order="F")
    Bb = np.array(fmu.getReal([vrs[f"Attachment_Point_Vectors[1,{i}]"] for i in range(1, 13)])).reshape((N, 3))
    m = fmu.getReal([vrs["Load_Mass"]])[0]
    L0 = fmu.getReal([vrs["Cable_Resting_Length"]])[0]
    return J, Bb, m, L0


def build_obs_rows(obs42, t):
    """Per-drone observations: 18 (load) + 6 (own drone pos/vel) + clock bank.
    The clock is common to all drones; the per-drone slot comes from own position."""
    load18 = obs42[0:18].copy()
    load18[3:12] = np.round(load18[3:12], 6)          # match _unpack_load's rounding of R
    clk = clock_features(t)
    rows = []
    for i in range(N):
        dpos = obs42[18 + 3 * i: 18 + 3 * i + 3]
        dvel = obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]
        rows.append(np.concatenate([load18, dpos, dvel, clk]).astype(np.float32))
    return rows


def collect():
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    prev_f = np.array([0.0, 0.0, FZ] * N)
    obs_rows, lam_rows = [], []

    t = 0.0
    while t < T_END - 1e-9:
        # obs + expert lambda, BOTH from the current state (no off-by-one).
        rows = build_obs_rows(obs42, t)
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]
        f_full, _, lam = agent.compute_forces(pos, vel, R, angvel, t, bypass_opt=BYPASS_OPT)
        for i in range(N):
            obs_rows.append(rows[i])
            lam_rows.append(lam[i])

        # LLC filter -> derivative -> step the plant.
        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    return (np.asarray(obs_rows, dtype=np.float32),
            np.asarray(lam_rows, dtype=np.float32).reshape(-1, 1))


if __name__ == "__main__":
    obs_arr, lam_arr = collect()
    np.savez(OUT, obs=obs_arr, lam=lam_arr)
    print(f"saved {OUT}")
    print(f"samples: {obs_arr.shape[0]}   obs {obs_arr.shape}   lam {lam_arr.shape}")
    print(f"lambda  range [{lam_arr.min():.3f}, {lam_arr.max():.3f}]  "
          f"mean {lam_arr.mean():.3f}  std {lam_arr.std():.3f}")
