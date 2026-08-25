"""
mappo.py — MAPPO / CTDE training of the F2 residual actor.

Shared residual actor  pi(delta_f | local obs 30-D)   -> the only thing that deploys.
Centralized critic     V(global true state 42-D)       -> training only (privileged).
Reward (ResidualMARLEnv): stay on the expert loiter loop + no stall + load guardrail,
all from the TRUE state. Cooperative -> TEAM reward = sum of per-drone rewards, one
GAE advantage stream shared by every agent (each agent contributes its own obs/action).

The actor starts fresh (~0 output = base-only, load-stable), so NO warm-start / anchor is
needed: RL explores from the safe base and learns the local feedback that keeps each drone
on its coordinated loop under desync. Run expert_reference.py first (-> expert_ref.npz).

Domain randomization: fresh sensing noise every reset (env RNG) + per-episode control
delays resampled from {1,2}. Saves residual_mappo.pt (actor) and residual_mappo_critic.pt.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from residual_marl_env import ResidualMARLEnv
from networks import Actor, Critic
from expert_reference import training_pairs, eval_scenarios
from collect_il_data import T_END          # 35 s — the horizon the trajectories + expert refs span

# --- desync spec (training distribution; matches the plan) ---
DESYNC = dict(pos_noise=0.03, vel_noise=0.10, noise_corr=0.995)
DELAY_CHOICES = (0, 2)

# ABLATION: True = delta_lambda-ONLY (zero the delta_wrench/load-trim head). Tests whether dw earns
# its keep or is just a load-disturbing stall crutch (removing it should drop swing toward base ~0.10).
DISABLE_DW = False

# DIAGNOSTIC: False = FIX one desync realization (deterministic env) to test whether the
# policy can overfit a SINGLE case at all. True = per-episode domain randomization (needs a
# big batch to average out draw variance -- that's what drowned the first run).
DOMAIN_RANDOMIZE = True
# TRAJECTORY randomization: per episode, sample a (reference, expert-path) pair from the training
# library (expert_lib.npz via training_pairs). True = generalize over the SAME 56-traj set F1 collected
# on; False = the single default trajectory (prior behavior). Needs `python expert_reference.py lib` first.
TRAJ_RANDOMIZE = True
GUARANTEE_ANCHOR = 2     # like F1's collect: force >= this many NON-quintic episodes/iter (default line +
                         #   solver-engaging customs). Bump to 2 for two. The rest are sampled from all 56.
FIXED_DELAYS = [1, 2, 2, 1]
FIXED_SEED = 12345
# Held-out deterministic eval scenario: a DIFFERENT seed than FIXED_SEED so DET_loop/DET_load
# score a case we did NOT overfit during the fixed-scenario pretrain.
EVAL_SEED = 4242
EVAL_DELAYS = [1, 2, 2, 1]

# --- PPO hyperparameters ---
ITERS = 150                  # warm-started from the fixed-scenario best -> generalizing, not learning
                             #   from scratch. ~88s/iter at 20k steps -> ~3.7h.
STEPS_PER_ITER = 35000       # ~10 episodes / update. Domain randomization adds per-SCENARIO draw variance
                             #   on top of sampling noise -> need more draws/update or the gradient thrashes.
REWARD_SCALE = 0.01          # scale raw rewards (~ -18000/ep) so critic targets are O(100); reporting stays RAW
EPOCHS = 10
MINIBATCH_STEPS = 512        # minibatch size in ENV STEPS (each expands to N agent samples)
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
ENT_COEF = 0.0               # REGIME 1: back to 0 (the original GitHub value). The entropy COLLAPSE that
                             #   forced ENT_COEF>0 was a REGIME-4 artifact of the JERK term (its penalty was
                             #   cheapest to cut by shrinking log_std -> killing exploration). With no jerk,
                             #   nothing pushes entropy down, so 0 works: PPO clip + the Gaussian's natural
                             #   exploration (LOG_STD_INIT=-1.0) suffice. (regime-4 sweep: 0.0015 floor,
                             #   0.003 crept up, 0.01 runaway -> all moot without jerk.)
MAX_GRAD = 1.0
LOG_STD_INIT = -1.0          # lower exploration (std~0.37) — std~0.6 kicks swamped the signal
HIDDEN = (128, 128)          # actor+critic width — OLD-NET arch (networks.py default), matches r4base.
                             #   Reverted from the (256,128,64) desired-state + leak experiment.
EVAL_EVERY = 4               # every N iters, eval the DETERMINISTIC (mean-action) policy on the 2-traj set
                             #   (line + held-out quintic) -> true loop_dist. Raised 2->4 so the 2-traj mean
                             #   costs the same total as the old 1-traj eval (representative selection, flat budget).
SEED = 0
DEVICE = "cpu"                        # tiny nets + sequential rollout -> CPU beats GPU (no per-step transfer)
WARMSTART = "residual_mappo_gt2_wide.pt"   # OLD-NET best (the -0.166 DET_R policy) — regime-2 warm-start.
                                         #   Arch/obs reverted to match it. CHANGE if you meant a different old ckpt.


def compute_gae(rew, val, done, gamma, lam):
    """GAE over a buffer that ends on episode boundaries (done=1 at each episode end)."""
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        nextv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nextv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    ret = adv + val[:T]
    return adv, ret


def estimate_norm(env, rng, pairs, n_traj=6):
    """Obs mean/std from BASE (zero-residual) rollouts across a SAMPLE of trajectories -- NOT just the
    default line. CRUCIAL for the DESIRED-STATE obs dims: they are EXACTLY constant on the line (y/z/roll/
    pitch/omega -> std~0), so line-only normalization made those dims ~1e6 on quintics (net garbage -> the
    quintic THRASH). Sampling quintics gives them real spread. ZERO action so episodes survive the FULL
    trajectory -> the MOVED desired-state (t~20-30s) is covered; sensing noise + base dynamics spread the
    rest. (Random actions would blow up early -> only cover t~0 -> wouldn't fix the moved-reference dims.)"""
    agents = env.possible_agents
    ad = env._act_space.shape[0]
    zero = {a: np.zeros(ad, np.float32) for a in agents}
    buf = []
    for _ in range(n_traj):
        env.traj, env.expert_pos = pairs[int(rng.integers(len(pairs)))] if pairs is not None \
            else (None, env.default_expert_pos)
        env.ctrl_delay = np.asarray(FIXED_DELAYS, dtype=int)
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        while env.agents:
            obs, *_ = env.step(zero)
            for a in agents:
                buf.append(obs[a])
    arr = np.asarray(buf, dtype=np.float32)
    return arr.mean(0, keepdims=True), (arr.std(0, keepdims=True) + 1e-6).astype(np.float32)


def eval_policy(env, actor, om, os_, scenarios):
    """Deterministic (mean-action) eval AVERAGED over `scenarios` = [(label, traj, expert_dpos)] =
    1 straight line + 1 HELD-OUT quintic. Score = MEAN over the set so best-net selection isn't a single
    lucky stick (and the held-out quintic makes it a generalization signal too). Same held-out desync
    (EVAL_SEED/EVAL_DELAYS) per traj. Returns mean metrics + per-traj loop (spread). Uses the mean action."""
    floor = env.epsilon + env.stall_margin              # cruise floor (below = stall risk)
    per = {k: [] for k in ("reward", "loop", "load", "loadmax", "swing", "vmin", "stallfrac", "coord", "jerk")}
    per_traj_loop = {}
    n_blow = 0                                          # deterministic (mean-policy) blowups -> a DEPLOYMENT hole
    for label, traj, epos in scenarios:
        env.traj, env.expert_pos = traj, epos           # (collect may have left another traj on the env)
        env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
        obs, _ = env.reset(seed=EVAL_SEED)
        agents = env.possible_agents
        loops, loads, swings, rews, speeds, coords, jerks = [], [], [], [], [], [], []
        k, blew = 0, False
        while env.agents:
            oa = np.stack([obs[a] for a in agents]).astype(np.float32)
            with torch.no_grad():
                mean = actor(torch.tensor(((oa - om) / os_).astype(np.float32), device=DEVICE)).cpu().numpy()
            obs, rewards, _, _, infos = env.step({a: mean[i] for i, a in enumerate(agents)})
            if infos[agents[0]].get("blowup"):           # guard fired -> the MEAN policy diverged this traj
                blew = True
            rews.append(np.mean([rewards[a] for a in agents]))   # per-step mean reward (the SELECTION metric)
            loops.append(np.mean([infos[a]["loop_dist"] for a in agents]))
            loads.append(infos[agents[0]]["load_err"])           # global (same in every agent's info)
            swings.append(infos[agents[0]]["load_verr"])
            coords.append(infos[agents[0]]["coord"])
            k += 1
            if k > env.stall_grace:                              # skip startup spin-up (not a stall)
                speeds.append(infos[agents[0]]["min_speed"])
                jerks.append(infos[agents[0]]["jerk"])
        speeds = np.asarray(speeds)
        vmin = float(speeds.min()) if speeds.size else 0.0       # empty if it blew before stall_grace
        # A DETERMINISTIC blowup is a deploy-fatal failure a late -100 among 1000s of clean steps would HIDE
        # in the mean -> DISQUALIFY the traj (sentinel reward/loop) so best-net selection can NEVER save it.
        if blew:
            n_blow += 1
            per["reward"].append(-50.0);                  per["loop"].append(5.0)
            per_traj_loop[label] = 5.0
        else:
            per["reward"].append(float(np.mean(rews)));   per["loop"].append(float(np.mean(loops)))
            per_traj_loop[label] = float(np.mean(loops))
        per["load"].append(float(np.mean(loads)));    per["loadmax"].append(float(np.max(loads)))
        per["swing"].append(float(np.mean(swings)));  per["vmin"].append(vmin)
        per["stallfrac"].append(float((speeds < floor).mean()) if speeds.size else 1.0)
        per["coord"].append(float(np.mean(coords)));  per["jerk"].append(float(np.mean(jerks)))
    # aggregate across the eval set: MEAN for the selection/tracking metrics; worst-case for the guards.
    out = dict(reward=float(np.mean(per["reward"])), loop=float(np.mean(per["loop"])),
               load=float(np.mean(per["load"])), loadmax=float(np.max(per["loadmax"])),
               swing=float(np.mean(per["swing"])), vmin=float(np.min(per["vmin"])),
               stallfrac=float(np.max(per["stallfrac"])), coord=float(np.mean(per["coord"])),
               jerk=float(np.mean(per["jerk"])), per_traj_loop=per_traj_loop, blowups=n_blow)
    return out


def critic_input(env):
    """PRIVILEGED critic state (CTDE, training-only): true global state (42) + the per-drone
    delay vector (n), 0-centered + the per-drone TIME-INDEXED expert TARGET (3n).
    The delays are a FIXED per-episode scenario parameter the decentralized actor never sees;
    giving them to the critic lets its value baseline explain away the scenario's difficulty.
    The TARGET is the crux for MULTI-TRAJECTORY: the reward is dominated by ||target - pos||^2,
    but state() alone can't say WHICH trajectory (the same load config maps to 56 different
    targets) -> V collapses -> the advantage is swamped by the traj DRAW, not the action. Handing
    V the phase-correct target (which it can difference against the drone positions in state())
    makes the value trajectory-aware, so the advantage isolates the ACTION. Training-only -> OK."""
    d = env.ctrl_delay.astype(np.float32) - float(np.mean(DELAY_CHOICES))
    idx = min(env._step, env.expert_pos.shape[1] - 1)          # current phase -> expert target point
    tgt = env.expert_pos[:, idx, :].reshape(-1).astype(np.float32)   # (3n,) per-drone target
    return np.concatenate([env.state().astype(np.float32), d, tgt])


def collect(env, actor, critic, n_steps, rng, om, os_, pairs, n_anchor):
    """Roll whole episodes until >= n_steps. Returns per-STEP buffers (obs has an N axis).
    Actor sees NORMALIZED obs ((obs-om)/os_); raw obs are stored (re-normalized in update).
    `pairs` = [(traj, expert_dpos)] training set; per episode we sample one (multi-traj). The FIRST
    GUARANTEE_ANCHOR episodes are drawn from the NON-quintic anchors (pairs[:n_anchor]) so every iter
    sees the default line + solver-engaging customs (like F1's collect), the rest from all 56."""
    obs_b, act_b, logp_b = [], [], []      # per step, shape (N, .)
    state_b, val_b, rew_b, done_b = [], [], [], []
    ep_rews, ep_loops = [], []
    n_blowups = 0                          # episodes the guard truncated (state diverged)

    agents = env.possible_agents
    steps, ep_i = 0, 0
    while steps < n_steps:
        # per-episode TRAJECTORY (multi-traj generalization): sample a (reference, expert-path) pair.
        # Guarantee the first GUARANTEE_ANCHOR episodes are NON-quintic anchors (pairs[:n_anchor]).
        if TRAJ_RANDOMIZE:
            hi = n_anchor if ep_i < GUARANTEE_ANCHOR else len(pairs)
            env.traj, env.expert_pos = pairs[int(rng.integers(hi))]
        else:
            env.traj, env.expert_pos = None, env.default_expert_pos
        # per-episode DESYNC (delays + noise realization).
        if DOMAIN_RANDOMIZE:
            env.ctrl_delay = rng.integers(DELAY_CHOICES[0], DELAY_CHOICES[1] + 1, size=env.n)
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        else:
            env.ctrl_delay = np.asarray(FIXED_DELAYS, dtype=int)     # fixed desync -> deterministic env
            obs, _ = env.reset(seed=FIXED_SEED)
        ep_r, ep_loop = 0.0, []
        while env.agents:
            obs_arr = np.stack([obs[a] for a in agents]).astype(np.float32)     # (N,30)
            state = critic_input(env)                                          # (42+n,) privileged
            with torch.no_grad():
                obs_n = ((obs_arr - om) / os_).astype(np.float32)
                dist = actor.distribution(torch.tensor(obs_n, device=DEVICE))
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)                          # (N,)
                value = float(critic(torch.tensor(state, device=DEVICE)))
            action = action.cpu().numpy(); logp = logp.cpu().numpy()
            acts = {a: action[i] for i, a in enumerate(agents)}
            nobs, rewards, term, trunc, infos = env.step(acts)

            obs_b.append(obs_arr); act_b.append(action); logp_b.append(logp)
            state_b.append(state); val_b.append(value)
            r_vec = np.array([rewards[a] for a in agents], dtype=np.float32)   # PER-DRONE rewards
            rew_b.append(REWARD_SCALE * r_vec)                               # (N,) scaled for GAE/critic
            done = bool(list(trunc.values())[0] or list(term.values())[0])
            done_b.append(1.0 if done else 0.0)

            ep_r += float(r_vec.sum())                                        # report RAW team
            ep_loop.append(np.mean([infos[a]["loop_dist"] for a in agents]))
            if infos[agents[0]].get("blowup"):
                n_blowups += 1                                                # guard fired (state diverged)
            obs = nobs
            steps += 1
        ep_rews.append(ep_r); ep_loops.append(np.mean(ep_loop))
        ep_i += 1

    return (np.array(obs_b), np.array(act_b), np.array(logp_b),
            np.array(state_b, dtype=np.float32), np.array(val_b, dtype=np.float32),
            np.array(rew_b, dtype=np.float32), np.array(done_b, dtype=np.float32),
            np.mean(ep_rews), np.mean(ep_loops), n_blowups)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    env = ResidualMARLEnv(**DESYNC, disable_dw=DISABLE_DW, end_time=T_END)   # 35 s: cover full trajs + hold
    N = env.n
    env.reset(seed=SEED)                 # populate the plant state so env.state() is valid
    env.default_expert_pos = env.expert_pos.copy()   # DEFAULT-traj expert ref (eval/estimate baseline)
    pairs, n_anchor = training_pairs() if TRAJ_RANDOMIZE else (None, 0)   # (traj, expert_dpos) per traj
    eval_scen = eval_scenarios()         # [(line), (held-out quintic)] -> mean = the SELECTION metric
    if TRAJ_RANDOMIZE:
        print(f"trajectory randomization ON: {len(pairs)} trajs ({n_anchor} non-quintic anchors, "
              f">= {GUARANTEE_ANCHOR}/iter guaranteed) from expert_lib.npz")
    print(f"eval on {len(eval_scen)} trajs {[s[0] for s in eval_scen]} (mean = best-net selection metric)")
    state_dim = critic_input(env).shape[0]        # 42 state + n delays + 3n expert target (see critic_input)
    obs_dim = env._obs_space.shape[0]
    act_dim = env._act_space.shape[0]         # 10 = delta_lambda(n=4) + delta_wrench(6)
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=HIDDEN).to(DEVICE)
    critic = Critic(state_dim=state_dim, hidden=HIDDEN).to(DEVICE)
    if WARMSTART:
        ck = torch.load(WARMSTART, map_location=DEVICE, weights_only=False)
        actor.load_state_dict(ck["state_dict"])
        om = ck["obs_mean"].astype(np.float32); os_ = ck["obs_std"].astype(np.float32)
        crit_msg = ""
        if ck.get("critic_state") is not None:               # warm critic too (avoids the value re-learn dip)
            try:
                critic.load_state_dict(ck["critic_state"]); crit_msg = " + critic"
            except RuntimeError:                             # dim changed (e.g. privileged-delays critic) -> reinit
                crit_msg = " + critic REINIT (state_dim changed)"
        print(f"actor warm-started from {WARMSTART} (+ its obs normalization{crit_msg})")
    else:
        om, os_ = estimate_norm(env, rng, pairs)   # base-rollout norm over SAMPLED trajs (covers desired-state)
        print(f"obs normalization estimated from a random rollout ({obs_dim}-D)")
    om_t = torch.tensor(om, device=DEVICE); os_t = torch.tensor(os_, device=DEVICE)
    actor.log_std.data.fill_(LOG_STD_INIT)      # set exploration scale (overrides warm-start's)
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    hist_R, hist_loop = [], []
    hist_det_it, hist_det = [], []
    best_reward = -float("inf")              # RESET each run: reward is comparable only within a fixed
                                             #   reward scheme, so "best since THIS run started" (not carried).

    def save_ckpt(path, best):               # full resumable state: actor + critic + norm + best reward
        torch.save({"state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
                    "critic_state": {k: v.cpu() for k, v in critic.state_dict().items()},
                    "obs_mean": om, "obs_std": os_, "obs_dim": obs_dim, "act_dim": act_dim,
                    "best_reward": best}, path)

    it = 0
    try:
      for it in range(1, ITERS + 1):
        t0 = time.perf_counter()
        (obs_b, act_b, logp_b, state_b, val_b, rew_b, done_b,
         mean_ep_r, mean_loop, n_blowups) = collect(env, actor, critic, STEPS_PER_ITER, rng, om, os_,
                                                     pairs, n_anchor)

        T = len(rew_b)
        advs = np.zeros((T, N), np.float32); rets = np.zeros((T, N), np.float32)
        for d in range(N):                    # PER-DRONE GAE against the shared value V(global)
            advs[:, d], rets[:, d] = compute_gae(rew_b[:, d], val_b, done_b, GAMMA, LAMBDA)
        adv = (advs - advs.mean()) / (advs.std() + 1e-8)     # (T,N)
        ret_mean = rets.mean(axis=1)                         # (T,) critic target = mean per-drone return

        tt = lambda a: torch.tensor(a, device=DEVICE)
        obs_t = tt(obs_b); act_t = tt(act_b); logp_old = tt(logp_b)           # (T,N,.)
        state_t = tt(state_b); adv_t = tt(adv); ret_t = tt(ret_mean)

        for _ in range(EPOCHS):
            idx = rng.permutation(T)
            for s in range(0, T, MINIBATCH_STEPS):
                mb = idx[s:s + MINIBATCH_STEPS]
                o = (obs_t[mb].reshape(-1, obs_dim) - om_t) / os_t             # (mb*N,obs_dim) normalized
                a = act_t[mb].reshape(-1, act_dim)
                lp_old = logp_old[mb].reshape(-1)
                A = adv_t[mb].reshape(-1)                                      # per-drone advantage

                dist = actor.distribution(o)
                lp = dist.log_prob(a).sum(-1)
                ratio = torch.exp(lp - lp_old)
                l_clip = -torch.min(ratio * A,
                                    torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A).mean()
                ent = dist.entropy().sum(-1).mean()
                loss_a = l_clip - ENT_COEF * ent
                opt_a.zero_grad(); loss_a.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD); opt_a.step()

                v = critic(state_t[mb])
                loss_c = ((v - ret_t[mb]) ** 2).mean()
                opt_c.zero_grad(); loss_c.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD); opt_c.step()

        dt = time.perf_counter() - t0
        hist_R.append(mean_ep_r); hist_loop.append(mean_loop)
        det_str = ""
        if it == 1 or it == ITERS or it % EVAL_EVERY == 0:
            e = eval_policy(env, actor, om, os_, eval_scen)
            hist_det_it.append(it); hist_det.append(e["loop"])
            spread = " ".join(f"{lbl[:4]} {v:.3f}" for lbl, v in e["per_traj_loop"].items())
            det_str = (f"  DET_R {e['reward']:.3f}  loop {e['loop']:.3f} [{spread}]  load {e['load']:.3f}"
                       f"  vmin {e['vmin']:.3f}  stall% {100 * e['stallfrac']:.1f}"
                       f"  swing {e['swing']:.3f}  coord {e['coord']:.3f}  jerk {e['jerk']:.3f}"
                       + (f"  DET_BLOWUPS {e['blowups']}" if e["blowups"] else ""))
            if e["reward"] > best_reward:   # select on DETERMINISTIC REWARD (encodes ALL objectives), not
                best_reward = e["reward"]   # DET_loop (blind to stall/load). Scoped to THIS run (reset above).
                save_ckpt("residual_mappo.pt", best_reward)
                det_str += f"  (new best {best_reward:.3f} -> saved)"
        blow_str = f"  blowups {n_blowups}" if n_blowups else ""
        print(f"iter {it:3d}  team_ep_R {mean_ep_r:9.2f}  sampled_loop {mean_loop:.3f}{det_str}  "
              f"| critic_loss {loss_c.item():.3f}  ent {ent.item():.3f}{blow_str}  | {dt:.1f}s")
    except KeyboardInterrupt:
        print(f"\n[interrupted at iter {it}] -> saving resume checkpoint")

    save_ckpt("residual_mappo_last.pt", best_reward)   # LATEST resumable state (resume via WARMSTART=this)
    env.close()
    print(f"best (deploy) -> residual_mappo.pt (BEST DET_R {best_reward:.3f});  resume -> residual_mappo_last.pt")

    # training curves
    its = range(1, len(hist_R) + 1)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(its, hist_R, "b-o", ms=3)
    ax[0].set_ylabel("team episode reward"); ax[0].grid(True)
    ax[1].plot(its, hist_loop, "r-o", ms=3, alpha=0.4, label="sampled (exploration)")
    ax[1].plot(hist_det_it, hist_det, "k-o", ms=4, label="deterministic (mean action)")
    ax[1].set_ylabel("mean loop dist (m)"); ax[1].set_xlabel("iteration"); ax[1].grid(True); ax[1].legend()
    fig.suptitle("MAPPO training — residual on manifold reward")
    plt.tight_layout(); plt.savefig("mappo_training.png", dpi=150)
    print("saved mappo_training.png")
    plt.show()


if __name__ == "__main__":
    main()
