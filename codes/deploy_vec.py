"""
deploy_vec.py — closed-loop test of the WHOLE-VECTOR policy (il_actor_vec.pt).

Each step: (load 18 + clock 14) -> lambda[n] -> f = G^+ w_d + N lambda -> LLC -> plant.
No per-drone obs, no reconstruction. The load is pinned by the wrench controller, so
the net's input distribution is stable -> should fly WITHOUT DAgger.
"""

import numpy as np
import torch

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from optimizer import cable_force_calculation
from controller import error_calculation, get_reference_trajectory
from networks import Actor
from collect_il_data import clock_features, read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ


def load_policy(path="il_actor_vec.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = Actor(obs_dim=ckpt["obs_mean"].shape[1], act_dim=N)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return net, ckpt["obs_mean"], ckpt["obs_std"]


def main():
    net, om, os_ = load_policy()
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)   # wrench controller only

    prev_f = np.array([0.0, 0.0, FZ] * N)
    load_hist, ref_hist = [], []
    dvel = [[] for _ in range(N)]

    t = 0.0
    while t < T_END - 1e-9:
        load18 = obs42[0:18].copy()
        load18[3:12] = np.round(load18[3:12], 6)
        X = ((np.concatenate([load18, clock_features(t)])[None, :] - om) / os_).astype(np.float32)
        with torch.no_grad():
            lam = net(torch.tensor(X)).numpy().flatten()       # (N,)

        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]
        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)
        f_full, _ = cable_force_calculation(R, Bb, w_d, lam, N)

        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()

        load_hist.append(pos.copy())
        ref_hist.append(get_reference_trajectory(t)[0].copy())
        for i in range(N):
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    load = np.array(load_hist); ref = np.array(ref_hist)
    err = np.linalg.norm(load - ref, axis=1)
    print(f"whole-vector deploy:  mean track {err.mean():.4f} m   max {err.max():.4f} m")
    for i in range(N):
        v = np.array(dvel[i])
        print(f"  drone {i+1}: vel min {v.min():.3f}  mean {np.mean(v):.3f}")
    print("(deploy_f1 was: mean 0.0813  max 0.3868  |  vel mean ~1.40-1.43)")


if __name__ == "__main__":
    main()
