"""
distill_test.py — CAPACITY vs PPO-OPTIMIZATION probe (supervised distillation).

The joint RL run couldn't fit {+y quintic, +x+y custom} together. Two candidates remain:
  (A) CAPACITY  — the net cannot REPRESENT both expert maps at once.
  (B) PPO OPT   — the map IS representable, but policy-gradient can't FIND it (exploration blows up).

Aliasing is already ruled out: the reference trajectory is in the observation, so the two datasets
live in DISJOINT obs regions -> the joint map is a genuine (separable) function. This script tests
REPRESENTABILITY directly, with PPO removed: roll each single-task EXPERT on its own trajectory to
harvest (obs -> expert mean-action) pairs, then SUPERVISED-fit ONE student net to both.

Verdict:
  joint MSE ~= solo MSE (per expert)  -> both representable together -> NOT capacity -> the wall is PPO.
  joint MSE >> solo MSE               -> genuine capacity/conflict -> a bigger net is the honest answer
                                         (WIDTHS sweep then shows whether more width closes the gap).

Run:  python distill_test.py
"""
import numpy as np
import torch
import torch.nn as nn

from residual_marl_env import ResidualMARLEnv
from networks import Actor
from controller import make_quintic_pose
from expert_reference import expert_path, training_pairs
from collect_il_data import T_END
from trajectories import BASE_POS, HOLD
from mappo import DESYNC, EVAL_DELAYS

# --- the two single-task experts + the trajectory each solved (horizon must match how it was trained) ---
Y_RAMP, Y_END = 16.0, HOLD + 16.0 + 2.0          # +y quintic: RUNG-1 gentle move, short 19 s horizon
EXPERTS = [
    ("yquintic",  "residual_mappo_overfit_ch_y.pt",        "quintic_y"),
    ("xy_custom", "residual_mappo_overfit_customxy_ch5.pt", "custom_xy"),
]

N_ROLL = 4                          # desync rollouts per expert (covers the obs distribution, not one path)
SEEDS = [4242, 7, 19, 101, 202, 303, 404, 505]   # first N_ROLL used
WIDTHS = [(256, 256), (512, 512)]   # student widths to try (current, then bigger — does width close a gap?)
EPOCHS = 400
BATCH = 1024
LR = 1e-3
DESYNC_CFG = DESYNC                  # distill over the SAME desync the policy trains under


def build_traj(kind):
    """Return (traj, expert_dpos, end_time) for a spec keyword."""
    if kind == "quintic_y":
        traj = make_quintic_pose(np.array([0.0, 1.0, 0.0]), np.zeros(3), ramp=Y_RAMP,
                                 hold=HOLD, base_pos=np.asarray(BASE_POS, float))
        dpos, _, _ = expert_path(traj, Y_END)
        return traj, dpos, Y_END
    if kind == "custom_xy":
        pr, _ = training_pairs()
        traj, dpos = pr[2]                      # +x+y custom, precomputed at T_END
        return traj, dpos, T_END
    raise ValueError(kind)


def load_actor(ckpt, env):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    om = ck["obs_mean"].astype(np.float32).reshape(-1); os_ = ck["obs_std"].astype(np.float32).reshape(-1)
    sd = ck["state_dict"]
    hidden = (sd["body.0.weight"].shape[0], sd["body.2.weight"].shape[0])
    actor = Actor(om.shape[0], env._act_space.shape[0], hidden=hidden)
    actor.load_state_dict(sd); actor.eval()
    return actor, om, os_


def harvest(name, ckpt, kind):
    """Roll the EXPERT (deterministic) on its trajectory over N_ROLL desync seeds; collect (raw obs,
    expert mean-action) per drone-step over the distribution the expert actually visits."""
    traj, dpos, end_time = build_traj(kind)
    env = ResidualMARLEnv(**DESYNC_CFG, end_time=end_time)
    actor, om, os_ = load_actor(ckpt, env)
    X, Y = [], []
    n_blew = 0
    for sd in SEEDS[:N_ROLL]:
        env.traj, env.expert_pos = traj, dpos
        env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
        obs, _ = env.reset(seed=sd)
        agents = env.possible_agents
        while env.agents:
            arr = np.stack([obs[a] for a in agents]).astype(np.float32)          # (n, obs)
            with torch.no_grad():
                mean = actor(torch.tensor((arr - om) / os_)).numpy()             # expert MEAN action (n, act)
            X.append(arr); Y.append(mean)
            obs, _, _, _, infos = env.step({a: mean[i] for i, a in enumerate(agents)})
            if infos[agents[0]].get("blowup"):
                n_blew += 1
                break
    env.close()
    X = np.concatenate(X, 0); Y = np.concatenate(Y, 0)
    print(f"  {name:<10} {X.shape[0]:>7} pairs   obs_dim {X.shape[1]}  act_dim {Y.shape[1]}"
          f"   (blew {n_blew}/{N_ROLL})", flush=True)
    return X, Y


def fit(X, Y, hidden, obs_dim, act_dim, groups=None, tag=""):
    """Supervised-fit a student net (MSE on the mean action). Returns per-group MSE and R^2."""
    om = X.mean(0).astype(np.float32); os_ = (X.std(0) + 1e-6).astype(np.float32)
    Xn = torch.tensor((X - om) / os_); Yt = torch.tensor(Y.astype(np.float32))
    student = Actor(obs_dim, act_dim, hidden=hidden)
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    M = X.shape[0]
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        idx = rng.permutation(M)
        for s in range(0, M, BATCH):
            mb = idx[s:s + BATCH]
            pred = student(Xn[mb])
            loss = ((pred - Yt[mb]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = student(Xn).numpy()
    se = ((pred - Y) ** 2)                                   # (M, act)
    out = {}
    for g in ([0, 1] if groups is not None else [None]):
        sel = np.ones(M, bool) if g is None else (groups == g)
        mse = float(se[sel].mean())
        var = float(Y[sel].var())
        out[g] = (mse, 1.0 - mse / (var + 1e-12))
    return student, om, os_, out


def main():
    print(f"Distillation capacity probe — {N_ROLL} rollouts/expert, widths {WIDTHS}, {EPOCHS} epochs\n")
    print("harvesting expert (obs -> mean-action) pairs:")
    data = [(name, *harvest(name, ckpt, kind)) for name, ckpt, kind in EXPERTS]
    obs_dim = data[0][1].shape[1]; act_dim = data[0][2].shape[1]

    Xa, Ya = data[0][1], data[0][2]
    Xb, Yb = data[1][1], data[1][2]
    Xj = np.concatenate([Xa, Xb], 0); Yj = np.concatenate([Ya, Yb], 0)
    gj = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])

    print("\nSOLO fit = the per-expert MSE FLOOR (one net, one trajectory). JOINT = one net, both.")
    print("If JOINT ~= SOLO -> both representable together -> NOT capacity (the wall is PPO).")
    print("If JOINT >> SOLO -> capacity/conflict (and see if a wider net closes it).\n")

    for hidden in WIDTHS:
        # solo floors
        _, _, _, sa = fit(Xa, Ya, hidden, obs_dim, act_dim)
        _, _, _, sb = fit(Xb, Yb, hidden, obs_dim, act_dim)
        solo = {0: sa[None], 1: sb[None]}
        # joint
        _, _, _, jo = fit(Xj, Yj, hidden, obs_dim, act_dim, groups=gj)

        print(f"hidden={hidden}", flush=True)
        print(f"  {'expert':<10}{'solo MSE':>12}{'joint MSE':>12}{'joint/solo':>12}{'solo R2':>10}{'joint R2':>10}")
        for g, (name, _, _) in zip((0, 1), EXPERTS):
            smse, sr2 = solo[g]; jmse, jr2 = jo[g]
            ratio = jmse / (smse + 1e-12)
            flag = "  <-- degraded" if ratio > 3 else ""
            print(f"  {name:<10}{smse:>12.5f}{jmse:>12.5f}{ratio:>12.2f}{sr2:>10.3f}{jr2:>10.3f}{flag}")
        print(flush=True)


if __name__ == "__main__":
    main()
