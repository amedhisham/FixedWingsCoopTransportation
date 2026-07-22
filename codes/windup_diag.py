"""
windup_diag.py — is the end-of-episode blow-up decentralized INTEGRAL WINDUP?

Runs ONE base-only desync episode (beta=0, no residual). Each drone's wrench_control
integrates its OWN noisy/delayed load error (AR(1) corr=0.995 = a persistent bias), so
intg_ep winds up -> the force part of w_d diverges -> the four drones' forces stop
cancelling -> the true load runs away (super-linear near the end).

A CENTRALIZED reference LocalModelAgent is fed the TRUE (noise-free) state every step; its
integrator sees the real (small) error, so it should stay bounded — exactly the beta=1
condition that held track at 0.0813. If the decentralized ||intg_ep|| / ||w_d_force||
ramp toward the end while the centralized ones stay flat, windup is the cause and
anti-windup on the wrench integrator is the fix. (The base NET only outputs lambda, which
is nullspace G.N=0, so it cannot touch this — that's why base training can't fix it.)
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from controller import get_reference_trajectory
from networks import Actor
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from collect_prdot_data import RECON_ALPHA
from residual_marl_env import LocalModelAgent
from dagger_noisy import DesyncSensor, DELAY_CHOICES, CLOCK_OFFSET

BASE_POLICY = "il_actor_prdot_dagger_analytic.pt"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main():
    ck = torch.load(BASE_POLICY, map_location="cpu", weights_only=False)
    base_net = Actor(obs_dim=ck["obs_mean"].shape[1], act_dim=N)
    base_net.load_state_dict(ck["state_dict"]); base_net.eval()
    om, os_ = ck["obs_mean"].astype(np.float32), ck["obs_std"].astype(np.float32)

    def net_lam(row):
        Xn = ((np.atleast_2d(row) - om) / os_).astype(np.float32)
        with torch.no_grad():
            return base_net(torch.tensor(Xn)).numpy()

    rng = np.random.default_rng(SEED)
    delays = rng.integers(DELAY_CHOICES[0], DELAY_CHOICES[1] + 1, size=N)
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    locals_ = [LocalModelAgent(N, DT, PHASES, EPS, L0, m, J, Bb, RECON_ALPHA) for _ in range(N)]
    ref = LocalModelAgent(N, DT, PHASES, EPS, L0, m, J, Bb, RECON_ALPHA)     # centralized, true-state
    sensor = DesyncSensor(N, delays, rng)

    prev_f = np.array([0.0, 0.0, FZ] * N)
    t = 0.0
    T = []
    intg = [[] for _ in range(N)]         # ||intg_ep|| per drone (position windup)
    intgR = [[] for _ in range(N)]        # ||intg_eR|| per drone (attitude windup)
    wd = [[] for _ in range(N)]           # ||force part of w_d|| per drone
    dvel = [[] for _ in range(N)]
    ff_norm = [[] for _ in range(N)]      # ||applied force|| per drone
    fdot = [[] for _ in range(N)]         # ||f_dot|| (the deriv handed to the FMU)
    gap = [[] for _ in range(N)]          # ||f_cmd - ff_prev|| (drives f_dot via the filter)
    tension = [[] for _ in range(N)]      # T_i = sqrt(||ff_i||^2 + 1e-6)
    amp = [[] for _ in range(N)]          # L0 / T_i  (the velocity amplification factor)
    ref_intg, ref_intgR, ref_wd = [], [], []
    track = []

    while t < T_END - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        # Decentralized: per-drone noisy/delayed views -> base lambda -> base-only drive.
        ests = sensor.update(pos, R, vel, angvel)
        rows = [locals_[i].prepare(p, v, Rr, w, t + CLOCK_OFFSET)
                for i, (p, Rr, v, w) in enumerate(ests)]
        lam = net_lam(np.stack(rows))                                       # (N,N)

        # Centralized reference on the TRUE state (integrator baseline; not driving the plant).
        rrow = ref.prepare(pos, vel, R, angvel, t)
        ref.finalize(net_lam(rrow)[0])

        f_applied = np.zeros(3 * N)
        for i in range(N):
            f_full = locals_[i].finalize(lam[i])
            sl = slice(3 * i, 3 * i + 3)
            f_applied[sl] = f_full[sl]
            intg[i].append(np.linalg.norm(locals_[i].intg_ep))
            intgR[i].append(np.linalg.norm(locals_[i].intg_eR))
            wd[i].append(np.linalg.norm(locals_[i]._w_d[:3]))              # force part of the wrench
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))

        ref_intg.append(np.linalg.norm(ref.intg_ep))
        ref_intgR.append(np.linalg.norm(ref.intg_eR))
        ref_wd.append(np.linalg.norm(ref._w_d[:3]))
        track.append(np.linalg.norm(pos - get_reference_trajectory(t)[0]))
        T.append(t)

        ff = LLC_ALPHA * f_applied + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        for i in range(N):
            sl = slice(3 * i, 3 * i + 3)
            Ti = float(np.sqrt(ff[sl] @ ff[sl] + 1e-6))
            ff_norm[i].append(float(np.linalg.norm(ff[sl])))
            fdot[i].append(float(np.linalg.norm(deriv[sl])))
            gap[i].append(float(np.linalg.norm(f_applied[sl] - prev_f[sl])))
            tension[i].append(Ti)
            amp[i].append(L0 / Ti)
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    T = np.array(T)

    # Fig 1: position integrator windup — decentralized vs centralized reference.
    plt.figure(figsize=(11, 5))
    for i in range(N):
        plt.plot(T, intg[i], label=f"drone {i+1} (noisy)")
    plt.plot(T, ref_intg, "k--", lw=2, label="centralized (true state)")
    plt.xlabel("Time (s)"); plt.ylabel(r"$\|\int e_p\,dt\|$")
    plt.title("Position integrator — windup check"); plt.legend(); plt.grid(True)

    # Fig 2: force part of w_d — decentralized vs centralized.
    plt.figure(figsize=(11, 5))
    for i in range(N):
        plt.plot(T, wd[i], label=f"drone {i+1} (noisy)")
    plt.plot(T, ref_wd, "k--", lw=2, label="centralized (true state)")
    plt.xlabel("Time (s)"); plt.ylabel(r"$\|f_{L,d}\|$  (force part of $w_d$)")
    plt.title("Desired load force — divergence check"); plt.legend(); plt.grid(True)

    # Fig 3: the symptom — load drift + drone velocity norms.
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(T, track, "b"); ax[0].set_ylabel("load track err (m)"); ax[0].grid(True)
    for i in range(N):
        ax[1].plot(T, dvel[i], label=f"drone {i+1}")
    ax[1].axhline(EPS, ls="--", c="gray", label="epsilon")
    ax[1].set_ylabel("velocity norm (m/s)"); ax[1].set_xlabel("Time (s)")
    ax[1].legend(); ax[1].grid(True)
    fig.suptitle("Symptom: load drift + velocity explosion")

    # Fig 4: velocity-explosion anatomy — is it f_dot (gap) or the L0/T (tension) factor?
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for i in range(N):
        ax[0].plot(T, dvel[i], label=f"drone {i+1}")
        ax[1].plot(T, fdot[i])
        ax[2].plot(T, tension[i])
        ax[3].plot(T, amp[i])
    ax[0].set_ylabel("||v_i|| (m/s)"); ax[0].legend(ncol=N, fontsize=8)
    ax[1].set_ylabel(r"$\|\dot f_i\|$")
    ax[2].set_ylabel(r"tension $T_i$")
    ax[3].set_ylabel(r"$L_0/T_i$ (amp)"); ax[3].set_xlabel("Time (s)")
    for a in ax:
        a.grid(True)
    fig.suptitle("Velocity-explosion anatomy: v vs f_dot vs tension vs amplification")

    # Pinpoint the worst velocity moment and decompose it.
    varr = np.array(dvel)                          # (N, steps)
    i_star, k_star = np.unravel_index(np.argmax(varr), varr.shape)
    print(f"\nvmax = {varr[i_star, k_star]:.2f} m/s  at drone {i_star+1}, t={T[k_star]:.2f}s")
    print(f"  at that instant:  ||f_dot|| {fdot[i_star][k_star]:.2f}   "
          f"gap ||f_cmd-ff|| {gap[i_star][k_star]:.3f}   "
          f"tension T {tension[i_star][k_star]:.4f}   L0/T {amp[i_star][k_star]:.2f}")
    print(f"  episode ranges (drone {i_star+1}):  T in "
          f"[{min(tension[i_star]):.4f}, {max(tension[i_star]):.3f}]   "
          f"L0/T max {max(amp[i_star]):.1f}   ||f_dot|| max {max(fdot[i_star]):.2f}")

    # Episode-wide: is v ugly even at NORMAL tension? Split velocity by tension regime.
    Tarr = np.array(tension); fdarr = np.array(fdot); gaparr = np.array(gap); amparr = np.array(amp)
    print("\nper-drone medians over episode:")
    for i in range(N):
        print(f"  drone {i+1}:  gap {np.median(gaparr[i]):.3f}  ||f_dot|| {np.median(fdarr[i]):.2f}  "
              f"L0/T {np.median(amparr[i]):.2f}  ||v|| {np.median(varr[i]):.2f}")
    hi_T = Tarr > 1.0          # comfortably-tensioned steps (no L0/T blow-up)
    lo_T = Tarr < 0.5          # near-slack steps
    v_hiT = varr[hi_T]; v_loT = varr[lo_T]
    print(f"\nvelocity by tension regime (all drones/steps):")
    print(f"  T>1.0 (normal, {hi_T.mean()*100:.0f}% of steps):  "
          f"||v|| median {np.median(v_hiT):.2f}  p95 {np.percentile(v_hiT, 95):.2f}  max {v_hiT.max():.2f}")
    if lo_T.any():
        print(f"  T<0.5 (slack,  {lo_T.mean()*100:.0f}% of steps):  "
              f"||v|| median {np.median(v_loT):.2f}  p95 {np.percentile(v_loT, 95):.2f}  max {v_loT.max():.2f}")
    print(f"  clean-loiter reference ~1.4 m/s")

    # Numeric: end-of-episode ramp (last 10%) vs centralized.
    tail = slice(int(0.9 * len(T)), None)
    dec_intg = float(np.mean([np.mean(np.array(intg[i])[tail]) for i in range(N)]))
    dec_wd = float(np.mean([np.mean(np.array(wd[i])[tail]) for i in range(N)]))
    print(f"delays {delays.tolist()}")
    print(f"tail (last 10%)  ||intg_ep||: decentralized {dec_intg:.3f}  vs  "
          f"centralized {np.mean(np.array(ref_intg)[tail]):.3f}  "
          f"(ratio {dec_intg / (np.mean(np.array(ref_intg)[tail]) + 1e-9):.1f}x)")
    print(f"tail (last 10%)  ||f_L,d||  : decentralized {dec_wd:.3f}  vs  "
          f"centralized {np.mean(np.array(ref_wd)[tail]):.3f}  "
          f"(ratio {dec_wd / (np.mean(np.array(ref_wd)[tail]) + 1e-9):.1f}x)")
    print(f"final load track err {track[-1]:.3f} m  (start {track[0]:.3f})")
    plt.show()


if __name__ == "__main__":
    main()
