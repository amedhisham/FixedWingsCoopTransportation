"""
analyze_lambda.py — expert vs policy lambda, and the residual's temporal structure.

Drives the plant with the OPTIMIZER (so we stay on the expert's trajectory, the
training distribution) and evaluates the trained policy at each state. Then:
  - overlays expert vs policy lambda over time (per drone),
  - plots the residual (policy - expert) over time,
  - plots the residual autocorrelation and prints lag-1 autocorrelation.

Lag-1 autocorr near 0  => residual is high-frequency / uncorrelated step-to-step
(so low-pass smoothing would help). Near 1 => residual is slow/smooth.
Shows the figures; saves nothing.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from collect_il_data import build_obs_rows, read_params
from deploy_f1 import load_policy

N = 4
DT, T_END = 0.01, 25.0
EPS = 0.25
PHASES = np.array([0.0, np.pi / 2, 0.0, np.pi / 2])
LLC_ALPHA = DT / (0.2 + DT)
FZ = 0.7 * 9.81 / 4

# MUST match the BYPASS_OPT used to collect the policy's training data:
#   True  -> expert lambda is the fixed sinusoid (optimizer off), fast, no CasADi.
#   False -> expert lambda is the adaptive optimizer's (optimizer on), slower.
BYPASS_OPT = False


def autocorr(x, maxlag):
    x = x - x.mean()
    denom = np.sum(x * x)
    return np.array([np.sum(x[: len(x) - k] * x[k:]) / denom for k in range(maxlag + 1)])


def main():
    net, obs_mean, obs_std = load_policy()

    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)   # optimizer ON -> drives plant, gives expert lambda

    prev_f = np.array([0.0, 0.0, FZ] * N)
    t_hist = []
    exp = [[] for _ in range(N)]     # expert (optimizer) lambda
    pol = [[] for _ in range(N)]     # policy lambda evaluated on the same state

    t = 0.0
    while t < T_END - 1e-9:
        rows = build_obs_rows(obs42, t)
        X = (np.stack(rows) - obs_mean) / obs_std
        with torch.no_grad():
            plam = net(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]
        f_full, _, elam = agent.compute_forces(pos, vel, R, angvel, t, bypass_opt=BYPASS_OPT)

        t_hist.append(t)
        for i in range(N):
            exp[i].append(elam[i])
            pol[i].append(plam[i])

        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    t_hist = np.array(t_hist)
    exp = [np.array(e) for e in exp]
    pol = [np.array(p) for p in pol]
    res = [pol[i] - exp[i] for i in range(N)]

    print("residual (policy - expert lambda):")
    for i in range(N):
        r = res[i]
        lag1 = np.corrcoef(r[:-1], r[1:])[0, 1]
        print(f"  drone {i+1}:  RMSE {np.sqrt(np.mean(r**2)):.3f}   lag-1 autocorr {lag1:.3f}")

    # Fig 1: expert vs policy lambda.
    fig1, ax1 = plt.subplots(N, 1, figsize=(11, 9), sharex=True)
    for i, ax in enumerate(ax1):
        ax.plot(t_hist, exp[i], "k", lw=1.0, label="expert (optimizer)")
        ax.plot(t_hist, pol[i], "r", lw=1.0, alpha=0.8, label="policy")
        ax.set_ylabel(f"$\\lambda_{i+1}$"); ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    ax1[-1].set_xlabel("Time (s)"); fig1.suptitle("Expert vs policy lambda")

    # Fig 2: residual over time.
    fig2, ax2 = plt.subplots(N, 1, figsize=(11, 9), sharex=True)
    for i, ax in enumerate(ax2):
        ax.plot(t_hist, res[i], "b", lw=0.8)
        ax.set_ylabel(f"resid $\\lambda_{i+1}$"); ax.grid(True)
    ax2[-1].set_xlabel("Time (s)"); fig2.suptitle("Residual (policy - expert lambda)")

    # Fig 3: residual autocorrelation.
    maxlag = 60
    plt.figure()
    for i in range(N):
        plt.plot(range(maxlag + 1), autocorr(res[i], maxlag), label=f"drone {i+1}")
    plt.axhline(0, c="gray", lw=0.5)
    plt.xlabel("lag (steps)"); plt.ylabel("autocorrelation")
    plt.title("Residual autocorrelation (fast drop to ~0 => high-frequency)")
    plt.legend(); plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
