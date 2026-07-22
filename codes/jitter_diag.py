"""
jitter_diag.py — localize the desync jitter: reconstruction vs net/feedback.

Under desync the drone velocity norm goes jittery at ZERO residual. Where does the
jitter enter the chain?
    sensing noise -> w_d / reconstruct -> pR_dot (net INPUT)
                  -> lambda (net OUTPUT) -> f_base -> LLC -> plant velocity

We log three points per drone -- reconstructed pR_dot norm (net input), lambda (net
output), and true plant velocity norm -- for a COHERENT run and a DESYNC run, and
compare the step-to-step jitter (mean |d/step|). The ratio desync/coherent tells us
where the noise is amplified:
  - pR_dot ratio explodes first  -> finite-difference reconstruction amplifies noise
  - pR_dot ~ok but lambda jitters -> the net / lambda self-feedback loop is the culprit
"""

import numpy as np
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv

N = 4
EPSILON = 0.25
RECON_TAU = 0.1   # reconstruction filter time const (OUR estimate; plant LLC is 0.2)
DESYNC = dict(pos_noise=0.03, vel_noise=0.10, noise_corr=0.995, ctrl_delay=[0, 2, 2, 1])


def run_episode(**kwargs):
    """Zero-residual rollout; log per-drone pR_dot (input), lambda (output), true vel."""
    env = ResidualMARLEnv(n_carriers=N, epsilon=EPSILON, recon_tau=RECON_TAU, **kwargs)
    env.reset(seed=0)
    zero = {a: np.zeros(3, dtype=np.float32) for a in env.possible_agents}

    t_hist = []
    vel = [[] for _ in range(N)]
    lam = [[] for _ in range(N)]
    prd = [[] for _ in range(N)]
    while env.agents:
        _, _, _, _, infos = env.step(zero)
        s = env.state()
        t_hist.append(env.t)
        for i in range(N):
            vel[i].append(np.linalg.norm(s[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
            lam[i].append(infos[f"drone_{i}"]["lambda"])
            prd[i].append(infos[f"drone_{i}"]["prdot_own"])
    env.close()
    return dict(t=np.array(t_hist),
                vel=[np.array(v) for v in vel],
                lam=[np.array(l) for l in lam],
                prd=[np.array(p) for p in prd])


def jitter(sig_list):
    """mean over drones of mean |step-to-step change| — a high-frequency-content proxy."""
    return float(np.mean([np.mean(np.abs(np.diff(s))) for s in sig_list]))


if __name__ == "__main__":
    coh = run_episode()
    des = run_episode(**DESYNC)

    print(f"recon_tau = {RECON_TAU}  (plant LLC = 0.2)")
    print("jitter (mean |d/step|), coherent vs desync — where does it enter?")
    print(f"  {'signal':16s} {'coherent':>10s} {'desync':>10s} {'ratio':>8s}")
    for name, key in [("pR_dot (input)", "prd"), ("lambda (output)", "lam"),
                      ("velocity (plant)", "vel")]:
        jc, jd = jitter(coh[key]), jitter(des[key])
        print(f"  {name:16s} {jc:10.5f} {jd:10.5f} {jd / max(jc, 1e-9):7.1f}x")

    # Reconstruction fidelity: |recon pR_dot norm - true drone vel norm|, mean over drones.
    fid = lambda d: float(np.mean([np.mean(np.abs(d["prd"][i] - d["vel"][i])) for i in range(N)]))
    print(f"recon fidelity (mean |recon - true| m/s):  coherent {fid(coh):.4f}   desync {fid(des):.4f}")

    # Traces (drone 1), coherent vs desync, at each stage of the chain.
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, key, ylabel in zip(
        axes, ["prd", "lam", "vel"],
        ["pR_dot norm (net INPUT)", "lambda (net OUTPUT)", "true vel norm (plant)"],
    ):
        ax.plot(coh["t"], coh[key][0], "g", lw=1.0, label="coherent")
        ax.plot(des["t"], des[key][0], "r", lw=1.0, alpha=0.8, label="desync")
        ax.set_ylabel(ylabel); ax.grid(True); ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Where does the jitter enter? (drone 1)")
    plt.show()
