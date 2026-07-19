"""
val_collector.py — verify the lean collector == the wrapper-based collection.

Runs the fast single-expert collector and the (trusted) 4-expert ResidualMARLEnv
coherent collection, and compares the resulting (obs, lambda) arrays.
Faithful  ==>  differences at float32 noise level.
"""

import numpy as np
from collect_il_data import collect as collect_lean
from residual_marl_env import ResidualMARLEnv


def collect_wrapper():
    env = ResidualMARLEnv(n_carriers=4, epsilon=0.25)     # coherent
    obs, _ = env.reset(seed=0)
    zero = {a: np.zeros(3, dtype=np.float32) for a in env.possible_agents}
    obs_rows, lam_rows = [], []
    while True:
        next_obs, _, _, _, infos = env.step(zero)
        for name in env.possible_agents:
            obs_rows.append(obs[name])
            lam_rows.append(infos[name]["lambda"])
        obs = next_obs
        if not env.agents:
            break
    env.close()
    return (np.asarray(obs_rows, dtype=np.float32),
            np.asarray(lam_rows, dtype=np.float32).reshape(-1, 1))


if __name__ == "__main__":
    o_lean, l_lean = collect_lean()
    o_wrap, l_wrap = collect_wrapper()
    print(f"lean {o_lean.shape}   wrapper {o_wrap.shape}")
    print(f"max |obs_lean - obs_wrap|    = {np.abs(o_lean - o_wrap).max():.3e}")
    print(f"max |lambda_lean - lambda_wrap| = {np.abs(l_lean - l_wrap).max():.3e}")
