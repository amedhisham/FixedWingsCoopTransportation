"""
sanity_f2.py — zero-noise / zero-residual check of the network-based F2 env.

With all noise off and delta_f = 0, the four LocalModelAgents are coherent replicas
of the F1 policy, so ResidualMARLEnv should track the load like deploy_f1 did
(deploy_f1: mean 0.081 m, max 0.387 m, drone vel ~1.40-1.43 m/s). The only expected
difference is the ~1 cm neighbor-reconstruction error, so it should be CLOSE, not
bit-identical.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from residual_marl_env import ResidualMARLEnv
from controller import get_reference_trajectory

N = 4
env = ResidualMARLEnv()                       # defaults: zero noise, il_actor_prdot_dagger.pt
obs, infos = env.reset(seed=0)
zero = {a: np.zeros(3, dtype=np.float32) for a in env.possible_agents}

t_hist, load_hist, ref_hist = [], [], []
dpos = [[] for _ in range(N)]
dvel = [[] for _ in range(N)]
lam_hist = [[] for _ in range(N)]

DIAG = 25   # print the first few steps to catch a blowup
step = 0
step_time = 0.0   # accumulated env.step() time = the decentralized compute cost
while env.agents:
    _t0 = time.perf_counter()
    obs, rewards, term, trunc, infos = env.step(zero)
    step_time += time.perf_counter() - _t0
    s = env.state()                            # true 42-D plant state
    step += 1
    if step <= DIAG:
        lp = s[0:3]
        dp = [np.linalg.norm(s[18 + 3 * i: 18 + 3 * i + 3]) for i in range(N)]
        print(f"step {step:3d}  load=[{lp[0]:+.3f} {lp[1]:+.3f} {lp[2]:+.3f}]  "
              f"|drone_pos|={[f'{d:.2f}' for d in dp]}  nan={np.isnan(s).any()}")
    if np.isnan(s).any() or np.abs(s).max() > 1e4:
        print(f"  --> blew up at step {step}")
        break
    t_hist.append(env.t)
    load_hist.append(s[0:3].copy())
    ref_hist.append(get_reference_trajectory(env.t)[0].copy())
    for i in range(N):
        dpos[i].append(s[18 + 3 * i: 18 + 3 * i + 3].copy())
        dvel[i].append(np.linalg.norm(s[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        lam_hist[i].append(infos[f"drone_{i}"]["lambda"])
env.close()

if not load_hist:
    import sys; sys.exit(0)

load = np.array(load_hist); ref = np.array(ref_hist)
err = np.linalg.norm(load - ref, axis=1)
print(f"F2 env (zero noise, zero residual):")
print(f"  mean load tracking error = {err.mean():.4f} m   max = {err.max():.4f} m")
for i in range(N):
    v = np.array(dvel[i])
    print(f"  drone {i+1}: velocity norm min {v.min():.3f}  mean {np.mean(v):.3f}")
print(f"\n(deploy_prdot was: mean 0.0813  max 0.3868  |  vel mean ~1.40-1.43)")

# --- decentralized timing benchmark ---
# Each step runs N independent local replicas. Per replica: wrench + reconstruct all N
# pR_dot + one net inference + a FULL cable_force_calculation (of which each drone keeps
# only its own 3-D slice). That is the honest decentralized cost — it scales with the
# fleet (O(N) replicas, each O(N) reconstruction + force calc), unlike one central solve.
print(f"\ntiming: {step_time:.3f} s of compute over {step} steps  "
      f"({1000 * step_time / step:.3f} ms/step, {env.dt * step / step_time:.1f}x real-time)")
print(f"  = {N} local replicas/step, each = wrench + reconstruct({N} pR_dot) + net "
      f"+ full force calc (own slice applied)")

# --- plots (same panels as deploy_prdot) ---
t_hist = np.array(t_hist)
dpos = [np.array(p) for p in dpos]
dvel = [np.array(v) for v in dvel]
lam_hist = [np.array(l) for l in lam_hist]
EPS = env.epsilon

# 1. Load position tracking.
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
    ax.plot(t_hist, ref[:, k], "k--", lw=2, label="reference")
    ax.plot(t_hist, load[:, k], "b", label="F2 decentralized")
    ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
axes[2].set_xlabel("Time (s)"); fig.suptitle("Load tracking — F2 decentralized (zero noise)")

# 2. Drone velocity norms.
plt.figure()
for i in range(N):
    plt.plot(t_hist, dvel[i], label=f"Drone {i+1}")
plt.axhline(EPS, ls="--", c="gray", label="epsilon")
plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
plt.title("Drone velocity norms — F2 decentralized"); plt.legend(); plt.grid(True)

# 3. Drone XY trajectories.
plt.figure(figsize=(8, 6))
for i in range(N):
    plt.plot(dpos[i][:, 0], dpos[i][:, 1], label=f"Drone {i+1}")
plt.plot(load[:, 0], load[:, 1], "k--", lw=2, label="Load")
plt.xlabel("X (m)"); plt.ylabel("Y (m)")
plt.title("Drone XY trajectories — F2 decentralized"); plt.legend(); plt.grid(True); plt.axis("equal")

# 4. Per-drone lambda (each drone's own slice, from its local replica).
fig4, ax4 = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
for i, ax in enumerate(ax4):
    ax.plot(t_hist, lam_hist[i], "m")
    ax.set_ylabel(f"$\\lambda_{i+1}$"); ax.grid(True)
ax4[-1].set_xlabel("Time (s)"); fig4.suptitle("Per-drone lambda — F2 decentralized (own slice)")

plt.show()
