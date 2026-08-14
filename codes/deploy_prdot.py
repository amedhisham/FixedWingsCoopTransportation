"""
deploy_prdot.py — closed-loop test of the pR_dot policy (il_actor_prdot.pt).

Each step mirrors collection EXACTLY (same reconstruct/build_input helpers, same
history bookkeeping), the only change being net-in-the-loop instead of the optimizer:

    pR_dot = reconstruct(load, w_d, lambda_{t-1}, lambda_dot_{t-1})   # optimizer-exact
    lambda = net([clock, pR_dot, lambda_{t-1}])
    f      = G^+ w_d + N lambda   ->  LLC  ->  plant

Because the input is built the same way it was during collection (optimizer-exact,
analytic pR_dot, lagged lambda), there is no train/deploy distribution mismatch. No
absolute-time memorization: the only time signal is the clock's phase.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from optimizer import cable_force_calculation
from controller import error_calculation, get_reference_trajectory
from networks import Actor
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from collect_prdot_data import Reconstructor, build_input, LAM0, SUFFIX
from trajectories import heldout_set, showcase_set

POLICY = f"il_actor_prdot_dagger{SUFFIX}.pt"   # the FINAL F1 policy (daggered, ANALYTIC-aware).
                                               #   was hardcoded "il_actor_prdot.pt" -> loaded a STALE
                                               #   July non-analytic net -> spurious "held-out garbage".

EVAL_HELDOUT = True   # __main__: True -> generalization test over HELD-OUT trajectories (never trained
                      #   on); False -> single run on the default straight-line trajectory (with plots).
HELDOUT_N = 10        # number of held-out trajectories to evaluate

SHOWCASE = "short"       # __main__ (overrides EVAL_HELDOUT): None | "short" | "long". Runs the demo
                      #   trajectories (trajectories.showcase_set) at their OWN lengths -> shows the
                      #   net handles any horizon. "short"=compact 25s, "long"=35s training-like.
SHOWCASE_IDX = 1   # None -> the WHOLE preset (line + all quintics); or an index to isolate ONE
                      #   (0=line, 1..M=quintics) — symmetric with deploy_compare.SHOWCASE_IDX.
SHOWCASE_M = 3        # quintics per showcase preset

# OUTPUT-side lambda low-pass (None = off). Applied lambda = a*lam_raw + (1-a)*applied_prev,
# a = DT/(tau+DT); the SMOOTHED lambda drives the force AND feeds back as lambda_{t-1}. Kills
# the >~1 Hz self-fed ripple that lambda_dot amplifies into the pR_dot spike-feedback buzz,
# while the 0.33 Hz loiter passes (~2% attenuation / ~12 deg lag at tau=0.1). Load-safe: lambda
# is a nullspace coordinate (G.N=0), so ANY lambda (filtered or not) leaves load tracking intact
# -> effectively a rate-limiter on the internal-force coordination. Validated in diagnose_buzz:
# buzz 0.31->0.016, pR_dot max 105->19. tau ~ 0.05-0.1 (lighter protects loiter vmin).
# DEMOTED to a final augment (post-hoc filtering detunes the loiter -> wacky lambda); the
# primary buzz fix is better/bigger training + hardness curation. Re-enable only trained-in-loop.
LAM_LP_TAU = 0.01


def load_policy(path=POLICY):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hidden = tuple(ckpt.get("hidden", (128, 128)))            # net size is self-describing in the ckpt
    net = Actor(obs_dim=ckpt["obs_mean"].shape[1], act_dim=N, hidden=hidden)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return net, ckpt["obs_mean"], ckpt["obs_std"]


def run_episode(net, om, os_, env, agent, Bb, L0, traj=None, t_end=None):
    """One closed-loop policy episode on reference `traj` (None -> default straight line).
    env + agent are REUSED across trajectories (reset here). Returns a history/metrics dict.
    `t_end` overrides the loop length (default T_END) so a demo can run ANY horizon (env's
    end_time must be >= t_end)."""
    t_end = T_END if t_end is None else t_end
    obs42, _ = env.reset()
    agent.reset()

    prev_f = np.array([0.0, 0.0, FZ] * N)
    prev_lam = LAM0.copy()
    lam_lp = LAM0.copy()                              # output-side low-pass state
    lam_a = None if LAM_LP_TAU is None else DT / (LAM_LP_TAU + DT)
    recon = Reconstructor(Bb, L0, DT)
    t_hist, load_hist, ref_hist = [], [], []
    dpos = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    lam_hist = [[] for _ in range(N)]

    t = 0.0
    loop_t0 = time.perf_counter()
    while t < t_end - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        # Same input construction as collection.
        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t, traj)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)
        vR = recon(R, vel, angvel, w_d, prev_lam)
        X = ((build_input(t, vR, prev_lam)[None, :] - om) / os_).astype(np.float32)
        with torch.no_grad():
            lam = net(torch.tensor(X)).numpy().flatten()       # (N,)

        if lam_a is not None:                                  # output-side low-pass (buzz fix)
            lam_lp = lam_a * lam + (1.0 - lam_a) * lam_lp
            lam = lam_lp.copy()                                # drives force AND feeds back

        f_full, _ = cable_force_calculation(R, Bb, w_d, lam, N)
        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()

        t_hist.append(t)
        load_hist.append(pos.copy())
        ref_hist.append(get_reference_trajectory(t, traj)[0].copy())
        for i in range(N):
            dpos[i].append(obs42[18 + 3 * i: 18 + 3 * i + 3].copy())
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
            lam_hist[i].append(lam[i])
        obs42, *_ = env.step(np.concatenate([ff, deriv]))

        recon.roll(lam)
        prev_lam = lam.copy()
        t += DT
    loop_time = time.perf_counter() - loop_t0

    load = np.array(load_hist); ref = np.array(ref_hist)
    err = np.linalg.norm(load - ref, axis=1)
    dvel = [np.array(v) for v in dvel]
    return dict(t=np.array(t_hist), load=load, ref=ref,
                dpos=[np.array(p) for p in dpos], dvel=dvel,
                lam=[np.array(l) for l in lam_hist],
                track_mean=float(err.mean()), track_max=float(err.max()),
                vmin=float(min(v.min() for v in dvel)), loop_time=loop_time)


def plot_episode(h, title=""):
    """The 4 diagnostic plots + numeric summary for one episode history dict."""
    t_hist, load, ref = h["t"], h["load"], h["ref"]
    dpos, dvel, lam_hist = h["dpos"], h["dvel"], h["lam"]
    sfx = f" — {title}" if title else ""

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(t_hist, ref[:, k], "k--", lw=2, label="reference")
        ax.plot(t_hist, load[:, k], "b", label="policy")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)"); fig.suptitle("Load tracking — pR_dot policy" + sfx)

    plt.figure()
    for i in range(N):
        plt.plot(t_hist, dvel[i], label=f"Drone {i+1}")
    plt.axhline(EPS, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title("Drone velocity norms — pR_dot policy" + sfx); plt.legend(); plt.grid(True)

    plt.figure(figsize=(8, 6))
    for i in range(N):
        plt.plot(dpos[i][:, 0], dpos[i][:, 1], label=f"Drone {i+1}")
    plt.plot(load[:, 0], load[:, 1], "k--", lw=2, label="Load")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("Drone XY trajectories — pR_dot policy" + sfx); plt.legend(); plt.grid(True); plt.axis("equal")

    fig4, ax4 = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(ax4):
        ax.plot(t_hist, lam_hist[i], "m")
        ax.set_ylabel(f"$\\lambda_{i+1}$ (action)"); ax.grid(True)
    ax4[-1].set_xlabel("Time (s)"); fig4.suptitle("Policy action (lambda) — pR_dot policy (closed loop)" + sfx)


def main(policy_path=POLICY, traj=None):
    """Single closed-loop run on `traj` (None -> default straight line) with the full plot set.
    Backward-compatible entry point (dagger calls main(path))."""
    net, om, os_ = load_policy(policy_path)
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)   # wrench controller only
    h = run_episode(net, om, os_, env, agent, Bb, L0, traj)
    env.close()

    n_steps = len(h["t"])
    print(f"sim loop: {h['loop_time']:.3f} s for {n_steps} steps  "
          f"({1000 * h['loop_time'] / n_steps:.3f} ms/step, {T_END / h['loop_time']:.1f}x real-time)")
    print(f"pR_dot deploy:  mean track {h['track_mean']:.4f} m   max {h['track_max']:.4f} m")
    for i in range(N):
        print(f"  drone {i+1}: vel min {h['dvel'][i].min():.3f}  mean {np.mean(h['dvel'][i]):.3f}")
    print("(deploy_vec/f1 was: mean 0.0813  max 0.3868  |  vel mean ~1.40-1.43)")
    plot_episode(h)
    plt.show()


def evaluate_heldout(policy_path=POLICY, M=HELDOUT_N, plot_first=True):
    """GENERALIZATION TEST: run the policy on M HELD-OUT trajectories (disjoint seed, never
    trained on). Reports per-trajectory + aggregate tracking; plots the first for eyeballing."""
    net, om, os_ = load_policy(policy_path)
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    print(f"HELD-OUT generalization test: {M} trajectories (never trained on)\n"
          f"{'traj':>4}  {'dpos':>22}  {'drot(deg)':>18}  {'track_mean':>10}  {'track_max':>9}  {'vmin':>6}")
    tms, tmx, vms, first = [], [], [], None
    for k, (traj, p) in enumerate(heldout_set(M)):
        h = run_episode(net, om, os_, env, agent, Bb, L0, traj)
        if first is None:
            first = h
        tms.append(h["track_mean"]); tmx.append(h["track_max"]); vms.append(h["vmin"])
        print(f"{k+1:>4}  {str(np.round(p['pos_delta'], 2)):>22}  "
              f"{str(np.round(np.rad2deg(p['rot_delta']), 1)):>18}  "
              f"{h['track_mean']:>10.4f}  {h['track_max']:>9.4f}  {h['vmin']:>6.3f}")
    env.close()

    print(f"\nAGGREGATE over {M} held-out:  track_mean {np.mean(tms):.4f} "
          f"(worst {np.max(tms):.4f})   track_max {np.max(tmx):.4f}   vmin {np.min(vms):.3f}")
    if plot_first and first is not None:
        plot_episode(first, title="held-out #1")
        plt.show()


def run_showcase(kind=SHOWCASE, M=SHOWCASE_M, policy_path=POLICY):
    """Run the net on the demo trajectories (any length) and plot each — thesis showcase.
    ONE env at end_time=T_END (the max horizon); each traj's own t_end just bounds the loop."""
    net, om, os_ = load_policy(policy_path)
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    demos = showcase_set(kind, M)
    if SHOWCASE_IDX is not None:
        demos = [demos[SHOWCASE_IDX]]          # isolate ONE (0=line, 1..M=quintics)
    print(f"SHOWCASE '{kind}': net (in-loop EMA) on demo trajectories — any horizon\n"
          f"{'label':>16}  {'t_end':>5}  {'track_mean':>10}  {'track_max':>9}  {'vmin':>6}")
    for label, traj, t_end in demos:
        h = run_episode(net, om, os_, env, agent, Bb, L0, traj, t_end=t_end)
        print(f"{label:>16}  {t_end:>5.0f}  {h['track_mean']:>10.4f}  {h['track_max']:>9.4f}  {h['vmin']:>6.3f}")
        plot_episode(h, title=f"{label}  (t_end={t_end:.0f}s)")
    env.close()
    plt.show()


if __name__ == "__main__":
    if SHOWCASE is not None:
        run_showcase()
    elif EVAL_HELDOUT:
        evaluate_heldout()
    else:
        main()
