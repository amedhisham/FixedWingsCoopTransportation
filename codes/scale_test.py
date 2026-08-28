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
from expert_reference import expert_path
from collect_il_data import T_END
from trajectories import BASE_POS, HOLD
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS

CKPT = "residual_mappo_wideh_ch2.pt"          # change to the policy you want to test
SCALES = [7.0]      # +x displacement (m)
PLOT_SCALES = [7.0]                 # which scale(s) to draw the usual per-run plots for
RAMP = 50.0                          # quintic move duration (s)
END_TIME = HOLD + RAMP + 1.0        # episode horizon: cover hold + full move + tail (was hard-capped at T_END=35!)
GRACE = 20
DESYNC_ON = True                    # False -> CLEAN plant: zero pos/vel noise + zero control delays
MOVE_DIR = (1.0, 1.0, 1.0)           # move DIRECTION; per-scale displacement = MOVE_DIR * SCALE (e.g. (0,1,0)=+y)
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
        satl.append(np.mean([infos[a]["sat_lam"] for a in agents]))
        satw.append(np.mean([infos[a]["sat_w"] for a in agents]))
    m = (np.mean(loop), np.mean(load), np.mean(satl), np.mean(satw)) if loop else (np.nan,) * 4
    return m + (blew, blow_loadoff, blow_vmax, blow_t), hist


def plot_run(hist, traj, eps, scale, mode=""):
    t = np.array(hist["t"])
    load = np.array(hist["load"])                       # (T,3)
    dpos = np.array(hist["dpos"])                       # (T,n,3)
    dvel = np.array(hist["dvel"])                       # (T,n)
    t_full = np.linspace(0.0, END_TIME, 500)           # FULL intended horizon (actual may die early)
    ref_full = np.array([traj(ti)[0] for ti in t_full])   # (·,3) intended load path to END_TIME
    died = t[-1] < END_TIME - 1.0                       # actual ended before the move finished -> blew up
    n = dpos.shape[1]
    tag = f"{scale:.0f} m {tuple(MOVE_DIR)} — {mode}" + (f"  (DIED @ {t[-1]:.0f}s)" if died else "")

    # 1. Load position tracking (xyz) — reference over FULL horizon vs actual (stops where it died)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, "XYZ")):
        ax.plot(t_full, ref_full[:, k], "k--", lw=2, label="reference (intended)")
        ax.plot(t, load[:, k], "b", label="load (actual)")
        if died:
            ax.axvline(t[-1], color="r", ls=":", lw=1.5, label="blew up")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
        ax.set_xlim(0, END_TIME)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Load position — {tag}")

    # 2. Drone velocity norms
    plt.figure()
    for i in range(n):
        plt.plot(t, dvel[:, i], label=f"Drone {i+1}")
    plt.axhline(eps, ls="--", c="gray", label="epsilon")
    if died:
        plt.axvline(t[-1], color="r", ls=":", lw=1.5, label="blew up")
    plt.xlim(0, END_TIME)
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms — {tag}"); plt.legend(); plt.grid(True)

    # 3. 3-D trajectories (drones + load)
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    for i in range(n):
        ax3.plot(dpos[:, i, 0], dpos[:, i, 1], dpos[:, i, 2], label=f"Drone {i+1}")
    ax3.plot(load[:, 0], load[:, 1], load[:, 2], "k--", lw=2, label="Load")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    ax3.set_title(f"Drone + load trajectories — {tag}"); ax3.legend()


def main():
    env = ResidualMARLEnv(**DESYNC_CFG, end_time=END_TIME, blowup_v=BLOWUP_V)
    actor, om, os_ = load_actor(env)
    print(f"scale test  ckpt={CKPT}  dir={MOVE_DIR}  desync={'ON' if DESYNC_ON else 'OFF (clean)'}"
          f"  (mean over episode, GRACE-skipped)\n")
    print(f"{'move(m)':<9}{'mode':<8}{'loop':>8}{'load':>8}{'sat_lam':>9}{'sat_w':>8}")
    for s in SCALES:
        traj = make_quintic_pose(np.array(MOVE_DIR, float) * s, np.zeros(3), RAMP, HOLD, np.asarray(BASE_POS, float))
        dpos, _, _ = expert_path(traj, END_TIME)
        do_plot = s in PLOT_SCALES
        lb, hist_b = roll(env, actor, om, os_, traj, dpos, use_policy=False, record=do_plot)
        lp, hist_p = roll(env, actor, om, os_, traj, dpos, use_policy=True, record=do_plot)
        def fmt(m):
            if not m[4]:
                return f"{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>9.2f}{m[3]:>8.2f}"
            if m[5] is None:
                return "   -- BLEW UP (NaN) --"
            return f"   BLEW @ {m[7]:.1f}s  load {m[5]:.2f}m off  vmax {m[6]:.0f} m/s"
        print(f"{s:<9.1f}{'base':<8}{fmt(lb)}")
        print(f"{'':<9}{'policy':<8}{fmt(lp)}")
        print()
        if do_plot:
            if hist_b and len(hist_b["t"]) > 1:
                plot_run(hist_b, traj, env.epsilon, s, "base")
            if hist_p and len(hist_p["t"]) > 1:
                plot_run(hist_p, traj, env.epsilon, s, "policy")
    env.close()
    plt.show()


if __name__ == "__main__":
    main()
