"""
val_wrapper.py — verify ResidualMARLEnv (zero noise, zero residual) == baseline.

With every disturbance off and delta_f = 0, the 4-expert stitch must collapse to
the single coherent controller. We compare the load trajectory against the
single-ClassicalAgent baseline from val_agent.run_agent().
Faithful  ==>  differences at float32 noise level.
"""

import numpy as np

from residual_marl_env import ResidualMARLEnv
from val_agent import run_agent   # single-ClassicalAgent baseline (returns forces, loads)


def run_wrapper():
    env = ResidualMARLEnv()                      # all noise defaults = 0
    env.reset()
    zero = {a: np.zeros(3, dtype=np.float32) for a in env.possible_agents}
    loads = []
    while True:
        loads.append(env.state()[0:3].copy())    # load pos at current t (pre-step)
        env.step(zero)
        if not env.agents:                        # truncated -> episode over
            break
    env.close()
    return np.array(loads)


if __name__ == "__main__":
    _, load_baseline = run_agent()
    load_wrapper = run_wrapper()

    n = min(len(load_baseline), len(load_wrapper))
    diff = np.abs(load_baseline[:n] - load_wrapper[:n]).max()
    print(f"baseline steps={len(load_baseline)}  wrapper steps={len(load_wrapper)}")
    print(f"max |load_baseline - load_wrapper| = {diff:.3e}")
