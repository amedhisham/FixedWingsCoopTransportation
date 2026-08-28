"""
direction_sweep.py — MAP the residual's failure surface over MOVE DIRECTION, on the interpretable
26-direction SIGN GRID: every combination of {+, 0, -} on (x,y,z) except (0,0,0):
    6 pure axes (+x, -x, +y, ...) + 12 edge-diagonals (+x+y, +x+z, ...) + 8 corners (+x+y+z, ...).
Each is unit-normalized then scaled to MAG. For each, rolls the DETERMINISTIC policy and records
mean loop_dist (sentinel 5.0 on blowup), mean sat_lam, blow flag, and the training COVERAGE near
that direction (# of the 50 quintics within 30 deg). Two things drop out:
  - WHERE it fails (loop),
  - WHICH failure mode (sat_lam >= 1 -> SATURATED = authority/straining; sat_lam < 1 while loop bad
    -> UNSATURATED = OOD/coverage), read against coverage.

Read-only: separate env + the CKPT below. Point CKPT at residual_mappo.pt (when idle) for the CURRENT
policy; a load rarely races the live run's save.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from residual_marl_env import ResidualMARLEnv
from networks import Actor
from controller import make_quintic_pose
from expert_reference import expert_path
from trajectories import BASE_POS, HOLD, train_set
from collect_prdot_data import N_TRAJ
from mappo import DESYNC, EVAL_SEED, EVAL_DELAYS
from collect_il_data import T_END

CKPT = "residual_mappo.pt"               # the CURRENT (live-best) policy; snapshot it first for zero race risk
MAG = 10.0                               # move magnitude (m) — well beyond training (~3 m) -> also magnitude-OOD
RAMP = 50.0                              # quintic move duration (s); peak vel ~1.875*MAG/RAMP ~ 0.375 m/s (gentle)
END_TIME = HOLD + RAMP + 1.0             # 2 + 50 + 1 = 53 s
GRACE = 20
COVER_DEG = 30.0                         # coverage radius: # training quintics within this angle of each dir


def sign_grid():
    """The 26 (sx,sy,sz) in {+1,0,-1}^3 minus (0,0,0), as (label, unit_dir), lexicographic in x,y,z."""
    ax = "xyz"
    out = []
    for sx in (1, 0, -1):
        for sy in (1, 0, -1):
            for sz in (1, 0, -1):
                s = (sx, sy, sz)
                if s == (0, 0, 0):
                    continue
                lbl = "".join(("+" if v > 0 else "-") + ax[k] for k, v in enumerate(s) if v != 0)
                d = np.array(s, float); d /= np.linalg.norm(d)
                out.append((lbl, d))
    return out


def load_actor(env):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    om = np.asarray(ck["obs_mean"], np.float32).reshape(-1)
    os_ = np.asarray(ck["obs_std"], np.float32).reshape(-1)
    sd = ck["state_dict"]
    hidden = (sd["body.0.weight"].shape[0], sd["body.2.weight"].shape[0])   # INFER width from the weights
    actor = Actor(obs_dim=om.shape[0], act_dim=env._act_space.shape[0], hidden=hidden)
    actor.load_state_dict(sd); actor.eval()
    return actor, om, os_


def roll(env, actor, om, os_, traj, dpos):
    """Deterministic rollout on one direction -> (mean_loop, mean_satL, blew)."""
    env.traj, env.expert_pos = traj, dpos
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
    obs, _ = env.reset(seed=EVAL_SEED)
    agents = env.possible_agents
    loops, sats, k, blew = [], [], 0, False
    while env.agents:
        arr = np.stack([obs[a] for a in agents]).astype(np.float32)
        with torch.no_grad():
            mean = actor(torch.tensor((arr - om) / os_)).numpy()
        obs, _, _, _, infos = env.step({a: mean[i] for i, a in enumerate(agents)})
        if infos[agents[0]].get("blowup"):
            blew = True; break
        k += 1
        if k > GRACE:
            loops.append(np.mean([infos[a]["loop_dist"] for a in agents]))
            sats.append(np.mean([infos[a]["sat_lam"] for a in agents]))
    if blew or not loops:
        return 5.0, float(np.mean(sats)) if sats else np.nan, True
    return float(np.mean(loops)), float(np.mean(sats)), False


def main():
    grid = sign_grid()
    env = ResidualMARLEnv(**DESYNC, end_time=END_TIME)
    actor, om, os_ = load_actor(env)

    # training-quintic unit directions (for coverage)
    Dq = np.array([p["pos_delta"] for _, p in train_set(N_TRAJ)])
    Uq = Dq / np.linalg.norm(Dq, axis=1, keepdims=True)
    cos_thresh = np.cos(np.radians(COVER_DEG))

    print(f"direction sweep  ckpt={CKPT}  mag={MAG}m  (det policy, EVAL_SEED + 0-1-2 delay walk)")
    print(f"timing: hold {HOLD}s  ramp {RAMP}s  end {END_TIME}s   |   {len(grid)} directions (sign grid)")
    print("PLAN (lexicographic in x,y,z; sign of each axis in {+,0,-}, 000 excluded):")
    for i, (lbl, d) in enumerate(grid):
        p = d * MAG
        print(f"  {i+1:2d}  {lbl:<8} -> ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})")
    print()

    rows = []
    print(f"{'#':>3} {'dir':<8}{'loop':>8}{'satL':>7}{'cover':>7}  result")
    for i, (lbl, d) in enumerate(grid):
        pos = d * MAG
        traj = make_quintic_pose(pos, np.zeros(3), RAMP, HOLD, np.asarray(BASE_POS, float))
        dpos, _, _ = expert_path(traj, max(END_TIME, T_END))
        loop, sat, blew = roll(env, actor, om, os_, traj, dpos)
        cover = int((Uq @ d > cos_thresh).sum())          # # training quintics within COVER_DEG of this dir
        rows.append(dict(lbl=lbl, loop=loop, sat=sat, blew=blew, cover=cover))
        tag = "BLEW UP" if blew else ("SATURATED" if sat >= 1.0 else "ok")
        print(f"{i+1:>3} {lbl:<8}{loop:>8.3f}{sat:>7.2f}{cover:>7d}  {tag}")
    env.close()

    # --- plot: sorted worst->best, loop bars (colored by regime) + coverage bars ---
    rows.sort(key=lambda r: r["loop"], reverse=True)
    labels = [r["lbl"] for r in rows]
    loops = np.array([r["loop"] for r in rows])
    covers = np.array([r["cover"] for r in rows])
    def color(r):
        if r["blew"]:
            return "black"                     # blowup
        if r["sat"] >= 1.0:
            return "crimson"                   # SATURATED -> authority/straining mode
        return "steelblue"                     # unsaturated -> OOD/coverage mode
    colors = [color(r) for r in rows]
    y = np.arange(len(rows))

    fig, (axL, axC) = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    axL.barh(y, loops, color=colors)
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=8)
    axL.invert_yaxis()                         # worst on top
    axL.set_xlabel("mean loop_dist  (5 = blowup)")
    axL.set_title(f"Failure by direction — {CKPT}  ({MAG}m)")
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(color="crimson", label="saturated (authority)"),
                        Patch(color="steelblue", label="unsaturated (OOD/coverage)"),
                        Patch(color="black", label="blowup")], loc="lower right", fontsize=8)
    axC.barh(y, covers, color="gray")
    axC.set_xlabel(f"# training quintics within {COVER_DEG:.0f} deg (coverage)")
    axC.set_title("Coverage per direction")
    for yi, c in zip(y, covers):
        axC.text(c + 0.1, yi, str(c), va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig("direction_sweep.png", dpi=120)     # always save (works headless); window still shows if interactive
    print("\nsaved figure -> direction_sweep.png")
    plt.show()


if __name__ == "__main__":
    main()
