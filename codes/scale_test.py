"""
scale_test.py — does the trained residual generalize ACROSS spatial scale? Trained quintics move
~1-3 m (POS_RANGE 3); this rolls the DETERMINISTIC policy on +x quintics of growing displacement
[1,3,5,10] m under desync and reports loop/load. base = zero residual (F1 only) for reference, so a
blow-up at 10 m is attributable to the BASE going out-of-distribution vs the RESIDUAL. The reference
obs (p_d up to 10 m) is far outside the training obs-norm range -> this directly probes the
scale-heterogeneity / normalization worry.

PLOTS (PLOT_SCALES): for each chosen scale, the usual per-run figures for the POLICY rollout:
load position xyz (vs reference), drone velocity norms (vs epsilon), and 3-D drone+load trajectories.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from residual_marl_env import ResidualMARLEnv
from networks import Actor
from controller import make_quintic_pose
from expert_reference import expert_path, training_pairs   # expert_path: MOVE_DIR quintics; training_pairs: custom lib
from collect_il_data import T_END
from trajectories import BASE_POS, HOLD
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS

CKPT = "residual_mappo.pt"          # change to the policy you want to test
SCALES = [7.0]      # +x displacement (m)
PLOT_SCALES = SCALES                # which scale(s) to draw the usual per-run plots for
RAMP = 23.0                          # quintic move duration (s)
END_TIME = HOLD + RAMP + 2        # episode horizon: cover hold + full move + tail (was hard-capped at T_END=35!)
GRACE = 20
DESYNC_ON = True                    # False -> CLEAN plant: zero pos/vel noise + zero control delays
MOVE_DIR = (1.0, -1.0, 0.3)           # move DIRECTION; per-scale displacement = MOVE_DIR * SCALE (e.g. (0,1,0)=+y)
USE_CUSTOM = False                   # True -> ignore MOVE_DIR/SCALES, test a custom_set() trajectory instead
CUSTOM_IDX = 0                      # which custom (const-velocity solver-engaging move): 0 +x, 1 +y, 2 +x+y,
                                     #   3 -x+y, 4 +x-y  (see trajectories.CUSTOM_VELS). Runs at its native T_END horizon.
DESYNC_CFG = DESYNC if DESYNC_ON else dict(pos_noise=0.0, vel_noise=0.0, noise_corr=0.0)
DELAYS = EVAL_DELAYS if DESYNC_ON else [0, 0, 0, 0]
BLOWUP_V = 1.0e6     # scale_test-ONLY divergence guard (env default 100). Raised so the EXPLOSION gets
                     #   RECORDED as a visible spike instead of truncating at ~100. Training env is separate
                     #   -> mappo keeps the default 100 and still truncates early; this does NOT affect it.


def load_actor(env):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    om = ck["obs_mean"].astype(np.float32).reshape(-1); os_ = ck["obs_std"].astype(np.float32).reshape(-1)
    obs_dim = om.shape[0]; act_dim = env._act_space.shape[0]
    sd = ck["state_dict"]
    hidden = (sd["body.0.weight"].shape[0], sd["body.2.weight"].shape[0])   # INFER width (128 or 256...)
    actor = Actor(obs_dim, act_dim, hidden=hidden)
    actor.load_state_dict(sd); actor.eval()
    return actor, om, os_


def roll(env, actor, om, os_, traj, dpos, use_policy, record=False):
    env.traj, env.expert_pos = traj, dpos
    env.ctrl_delay = np.asarray(DELAYS, dtype=int)
    obs, _ = env.reset(seed=EVAL_SEED)
    agents = env.possible_agents
    n = env.n
    loop, load, satl, satw = [], [], [], []
    loopsq, loadsq = [], []                       # squared per-step errors -> MSE (m^2) for comparisons
    hist = {"t": [], "load": [], "dpos": [], "dvel": []} if record else None
    k, blew = 0, False
    blow_t = blow_loadoff = blow_vmax = None
    while env.agents:
        if use_policy:
            arr = np.stack([obs[a] for a in agents]).astype(np.float32)
            with torch.no_grad():
                mean = actor.distribution(torch.tensor((arr - om) / os_)).mean.numpy()
            act = {a: mean[i] for i, a in enumerate(agents)}
        else:
            act = {a: np.zeros(env._act_space.shape[0], np.float32) for a in agents}
        obs, _, _, _, infos = env.step(act)
        if infos[agents[0]].get("blowup"):          # tension collapse -> state diverged
            blew = True
            sd = infos[agents[0]].get("blowup_state")     # the EXPLODED 42-D state (huge but usually finite)
            if sd is not None and np.isfinite(sd).all():
                blow_t = env.t
                blow_loadoff = float(np.linalg.norm(sd[0:3] - traj(env.t)[0]))   # load pos vs where it SHOULD be
                blow_vmax = float(np.max(np.linalg.norm(sd[18 + 3 * n:18 + 6 * n].reshape(n, 3), axis=1)))
                if record:                                 # append the explosion frame so the plot SHOWS it
                    hist["t"].append(env.t); hist["load"].append(sd[0:3].copy())
                    hist["dpos"].append(sd[18:18 + 3 * n].reshape(n, 3).copy())
                    hist["dvel"].append(np.linalg.norm(sd[18 + 3 * n:18 + 6 * n].reshape(n, 3), axis=1))
            break
        if record:
            s = env.state()
            hist["t"].append(env.t)
            hist["load"].append(s[0:3].copy())
            hist["dpos"].append(s[18:18 + 3 * n].reshape(n, 3).copy())
            hist["dvel"].append(np.linalg.norm(s[18 + 3 * n:18 + 6 * n].reshape(n, 3), axis=1))
        k += 1
        if k <= GRACE:
            continue
        loop.append(np.mean([infos[a]["loop_dist"] for a in agents]))
        load.append(np.mean([infos[a]["load_err"] for a in agents]))
        loopsq.append(np.mean([infos[a]["loop_dist"] ** 2 for a in agents]))   # per-drone squared -> MSE
        loadsq.append(np.mean([infos[a]["load_err"] ** 2 for a in agents]))
        satl.append(np.mean([infos[a]["sat_lam"] for a in agents]))
        satw.append(np.mean([infos[a]["sat_w"] for a in agents]))
    m = ((np.mean(loop), np.mean(load), np.mean(loopsq), np.mean(loadsq), np.mean(satl), np.mean(satw))
         if loop else (np.nan,) * 6)
    return m + (blew, blow_loadoff, blow_vmax, blow_t), hist


TIME_MARKS = 7         # 3-D trajectory plot: this many evenly-spaced "t=Xs" numbers along each DRONE path


def _mark_times(ax, t, xyz, n=TIME_MARKS, color="k", label=True):
    """Drop n evenly-spaced t=Xs markers along a 3-D (x,y,z) path so the spatial plot carries a time axis.
    xyz: (T,>=3) positions on the same time grid as t. Snaps each target time to the nearest sample."""
    if n <= 0 or len(t) < 2:
        return
    t = np.asarray(t)
    for tm in np.linspace(t[0], t[-1], n):
        k = min(int(np.argmin(np.abs(t - tm))), len(xyz) - 1)   # clamp (path may be 1 sample shorter)
        x, y, z = xyz[k, 0], xyz[k, 1], xyz[k, 2]
        ax.plot([x], [y], [z], "o", color=color, ms=4, mfc="white", mew=1.2, zorder=6)
        if label:
            ax.text(x, y, z, f" t={t[k]:.0f}s", fontsize=7, color=color, zorder=7)


def plot_run(hist, traj, eps, tag_label, mode="", end_time=None):
    if end_time is None:
        end_time = END_TIME
    t = np.array(hist["t"])
    load = np.array(hist["load"])                       # (T,3)
    dpos = np.array(hist["dpos"])                       # (T,n,3)
    dvel = np.array(hist["dvel"])                       # (T,n)
    t_full = np.linspace(0.0, end_time, 500)           # FULL intended horizon (actual may die early)
    ref_full = np.array([traj(ti)[0] for ti in t_full])   # (·,3) intended load path to end_time
    died = t[-1] < end_time - 1.0                       # actual ended before the move finished -> blew up
    n = dpos.shape[1]
    tag = f"{tag_label} — {mode}" + (f"  (DIED @ {t[-1]:.0f}s)" if died else "")

    # 1. Load position tracking (xyz) — reference over FULL horizon vs actual (stops where it died)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, "XYZ")):
        ax.plot(t_full, ref_full[:, k], "k--", lw=2, label="reference (intended)")
        ax.plot(t, load[:, k], "b", label="load (actual)")
        if died:
            ax.axvline(t[-1], color="r", ls=":", lw=1.5, label="blew up")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
        ax.set_xlim(0, end_time)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Load position — {tag}")

    # 2. Drone velocity norms
    plt.figure()
    for i in range(n):
        plt.plot(t, dvel[:, i], label=f"Drone {i+1}")
    plt.axhline(eps, ls="--", c="gray", label="epsilon")
    if died:
        plt.axvline(t[-1], color="r", ls=":", lw=1.5, label="blew up")
    plt.xlim(0, end_time)
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms — {tag}"); plt.legend(); plt.grid(True)

    # 3. 3-D trajectories (drones + load), with t=Xs numbers along each drone path
    s0 = 2                                               # skip the t=0 sample (weird init jump)
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    for i in range(n):
        ax3.plot(dpos[s0:, i, 0], dpos[s0:, i, 1], dpos[s0:, i, 2], color=f"C{i}", label=f"Drone {i+1}")
        _mark_times(ax3, t[s0:], dpos[s0:, i, :], color=f"C{i}", label=True)   # NUMBERS on drone paths
    ax3.plot(load[s0:, 0], load[s0:, 1], load[s0:, 2], "k--", lw=2, label="Load")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    ax3.set_title(f"Drone + load trajectories — {tag}"); ax3.legend()


def fmt(m):
    # m = (loop, load, loopMSE, loadMSE, sat_lam, sat_w, blew, blow_loadoff, blow_vmax, blow_t)
    if not m[6]:
        return f"{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>10.4f}{m[3]:>10.4f}{m[4]:>9.2f}{m[5]:>8.2f}"
    if m[7] is None:
        return "   -- BLEW UP (NaN) --"
    return f"   BLEW @ {m[9]:.1f}s  load {m[7]:.2f}m off  vmax {m[8]:.0f} m/s"


def main():
    horizon = T_END if USE_CUSTOM else END_TIME       # customs span [hold, hold+move_dur] at T_END
    env = ResidualMARLEnv(**DESYNC_CFG, end_time=horizon, blowup_v=BLOWUP_V)
    actor, om, os_ = load_actor(env)

    def run_one(traj, dpos, col_label, plot_tag, do_plot):
        lb, hist_b = roll(env, actor, om, os_, traj, dpos, use_policy=False, record=do_plot)
        lp, hist_p = roll(env, actor, om, os_, traj, dpos, use_policy=True, record=do_plot)
        print(f"{col_label:<9}{'base':<8}{fmt(lb)}")
        print(f"{'':<9}{'policy':<8}{fmt(lp)}")
        print()
        if do_plot:
            if hist_b and len(hist_b["t"]) > 1:
                plot_run(hist_b, traj, env.epsilon, plot_tag, "base", horizon)
            if hist_p and len(hist_p["t"]) > 1:
                plot_run(hist_p, traj, env.epsilon, plot_tag, "policy", horizon)

    if USE_CUSTOM:
        from trajectories import custom_set
        pairs, n_anchor = training_pairs()               # anchors (customs) precomputed in expert_lib.npz
        assert CUSTOM_IDX < n_anchor, f"CUSTOM_IDX {CUSTOM_IDX} >= {n_anchor} customs"
        ctraj, dpos = pairs[CUSTOM_IDX]                   # (traj, PRECOMPUTED dpos) — no expert rollout, no IPOPT
        name = custom_set()[CUSTOM_IDX][1]["name"]
        print(f"scale test  ckpt={CKPT}  CUSTOM[{CUSTOM_IDX}]={name}  desync={'ON' if DESYNC_ON else 'OFF (clean)'}"
              f"  (mean over episode, GRACE-skipped)\n")
        print(f"{'traj':<9}{'mode':<8}{'loop':>8}{'load':>8}{'loopMSE':>10}{'loadMSE':>10}{'sat_lam':>9}{'sat_w':>8}")
        run_one(ctraj, dpos, name, f"custom[{CUSTOM_IDX}] {name}", do_plot=True)
    else:
        print(f"scale test  ckpt={CKPT}  dir={MOVE_DIR}  desync={'ON' if DESYNC_ON else 'OFF (clean)'}"
              f"  (mean over episode, GRACE-skipped)\n")
        print(f"{'move(m)':<9}{'mode':<8}{'loop':>8}{'load':>8}{'loopMSE':>10}{'loadMSE':>10}{'sat_lam':>9}{'sat_w':>8}")
        for s in SCALES:
            traj = make_quintic_pose(np.array(MOVE_DIR, float) * s, np.zeros(3), RAMP, HOLD, np.asarray(BASE_POS, float))
            dpos, _, _ = expert_path(traj, END_TIME)
            run_one(traj, dpos, f"{s:.1f}", f"{s:.0f} m {tuple(MOVE_DIR)}", do_plot=(s in PLOT_SCALES))
    env.close()
    plt.show()


if __name__ == "__main__":
    main()
