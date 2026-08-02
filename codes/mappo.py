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

# --- desync spec (training distribution; matches the plan) ---
DESYNC = dict(pos_noise=0.03, vel_noise=0.10, noise_corr=0.995)
DELAY_CHOICES = (1, 2)

# ABLATION: True = delta_lambda-ONLY (zero the delta_wrench/load-trim head). Tests whether dw earns
# its keep or is just a load-disturbing stall crutch (removing it should drop swing toward base ~0.10).
DISABLE_DW = True

# DIAGNOSTIC: False = FIX one desync realization (deterministic env) to test whether the
# policy can overfit a SINGLE case at all. True = per-episode domain randomization (needs a
# big batch to average out draw variance -- that's what drowned the first run).
DOMAIN_RANDOMIZE = True
FIXED_DELAYS = [1, 2, 2, 1]
FIXED_SEED = 12345
# Held-out deterministic eval scenario: a DIFFERENT seed than FIXED_SEED so DET_loop/DET_load
# score a case we did NOT overfit during the fixed-scenario pretrain.
EVAL_SEED = 4242
EVAL_DELAYS = [1, 2, 2, 1]

# --- PPO hyperparameters ---
ITERS = 150                  # warm-started from the fixed-scenario best -> generalizing, not learning
                             #   from scratch. ~88s/iter at 20k steps -> ~3.7h.
STEPS_PER_ITER = 20000       # ~8 episodes / update. Domain randomization adds per-SCENARIO draw variance
                             #   on top of sampling noise -> need more draws/update or the gradient thrashes.
REWARD_SCALE = 0.01          # scale raw rewards (~ -18000/ep) so critic targets are O(100); reporting stays RAW
EPOCHS = 10
MINIBATCH_STEPS = 512        # minibatch size in ENV STEPS (each expands to N agent samples)
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
ENT_COEF = 0.0015            # entropy bonus. 0.0 -> collapse (ent 4->-0.2); 0.01 -> RUNAWAY (ent 4.25->5.4);
                             #   0.003 STILL crept up 4.5->4.9 while DET_R/load/swing all regressed (entropy
                             #   bonus out-pushing the tiny task gradient, critic_loss ~0). 0.0015 = floor
                             #   exploration without inflating it -> want ent to PLATEAU ~4.3, not creep up.
MAX_GRAD = 1.0
LOG_STD_INIT = -1.0          # lower exploration (std~0.37) — std~0.6 kicks swamped the signal
EVAL_EVERY = 2               # every N iters, eval the DETERMINISTIC (mean-action) policy -> true loop_dist
SEED = 0
DEVICE = "cpu"                        # tiny nets + sequential rollout -> CPU beats GPU (no per-step transfer)
WARMSTART = "residual_mappo.pt"       # BEST-DET_R checkpoint (the -0.166 policy), NOT _last: _last is the
                                       #   resume/last-step state, always PAST the peak when the policy is
                                       #   drifting -> resuming from it gives back the gain (cost us run 2).


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


def estimate_norm(env, rng, n_steps=2500):
    """Roll random actions on the fixed scenario to estimate obs mean/std. The 44-D obs spans
    very different scales (positions ~10, velocities ~40, clock ~1) -> normalization matters."""
    env.ctrl_delay = np.asarray(FIXED_DELAYS, dtype=int)
    obs, _ = env.reset(seed=FIXED_SEED)
    agents = env.possible_agents
    ad = env._act_space.shape[0]
    buf, steps = [], 0
    while steps < n_steps and env.agents:
        acts = {a: rng.normal(0, 0.3, size=ad).astype(np.float32) for a in agents}
        obs, *_ = env.step(acts)
        for a in agents:
            buf.append(obs[a])
        steps += 1
    arr = np.asarray(buf, dtype=np.float32)
    return arr.mean(0, keepdims=True), (arr.std(0, keepdims=True) + 1e-6).astype(np.float32)


def eval_policy(env, actor, om, os_):
    """Deterministic (mean-action, no sampling) rollout on a HELD-OUT scenario (EVAL_SEED, distinct
    from the training FIXED_SEED so we don't score the seed we overfit) -> the TRUE mean loop_dist
    AND mean load-tracking error, free of exploration noise. Uses actor.forward (the mean)."""
    env.ctrl_delay = np.asarray(EVAL_DELAYS, dtype=int)
    obs, _ = env.reset(seed=EVAL_SEED)
    agents = env.possible_agents
    loops, loads, swings, rews, speeds, coords, jerks = [], [], [], [], [], [], []
    floor = env.epsilon + env.stall_margin              # cruise floor (below = stall risk)
    k = 0
    while env.agents:
        oa = np.stack([obs[a] for a in agents]).astype(np.float32)
        with torch.no_grad():
            mean = actor(torch.tensor(((oa - om) / os_).astype(np.float32), device=DEVICE)).cpu().numpy()
        obs, rewards, _, _, infos = env.step({a: mean[i] for i, a in enumerate(agents)})
        rews.append(np.mean([rewards[a] for a in agents]))  # per-step mean reward (the SELECTION metric)
        loops.append(np.mean([infos[a]["loop_dist"] for a in agents]))
        loads.append(infos[agents[0]]["load_err"])          # load pos error is global (same for all drones)
        swings.append(infos[agents[0]]["load_verr"])        # load VELOCITY error = swing rate
        coords.append(infos[agents[0]]["coord"])            # ||sum internal force|| (coordination/leak)
        k += 1
        if k > env.stall_grace:                             # skip startup (drones spin up from rest, not a stall)
            speeds.append(infos[agents[0]]["min_speed"])    # slowest drone this step (stall monitor)
            jerks.append(infos[agents[0]]["jerk"])          # velocity jitter (skip the startup jolt)
    speeds = np.asarray(speeds)
    # mean-per-step reward (length-robust: a blowup -> high per-step penalty, not rewarded for a short ep)
    return dict(reward=float(np.mean(rews)), loop=float(np.mean(loops)),
                load=float(np.mean(loads)), loadmax=float(np.max(loads)), swing=float(np.mean(swings)),
                vmin=float(speeds.min()), stallfrac=float((speeds < floor).mean()),
                coord=float(np.mean(coords)), jerk=float(np.mean(jerks)))


def collect(env, actor, critic, n_steps, rng, om, os_):
    """Roll whole episodes until >= n_steps. Returns per-STEP buffers (obs has an N axis).
    Actor sees NORMALIZED obs ((obs-om)/os_); raw obs are stored (re-normalized in update)."""
    obs_b, act_b, logp_b = [], [], []      # per step, shape (N, .)
    state_b, val_b, rew_b, done_b = [], [], [], []
    ep_rews, ep_loops = [], []
    n_blowups = 0                          # episodes the guard truncated (state diverged)

    agents = env.possible_agents
    steps = 0
    while steps < n_steps:
        if DOMAIN_RANDOMIZE:
            env.ctrl_delay = rng.integers(DELAY_CHOICES[0], DELAY_CHOICES[1] + 1, size=env.n)
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        else:
            env.ctrl_delay = np.asarray(FIXED_DELAYS, dtype=int)     # fixed desync -> deterministic env
            obs, _ = env.reset(seed=FIXED_SEED)
        ep_r, ep_loop = 0.0, []
        while env.agents:
            obs_arr = np.stack([obs[a] for a in agents]).astype(np.float32)     # (N,30)
            state = env.state().astype(np.float32)                             # (42,)
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

    return (np.array(obs_b), np.array(act_b), np.array(logp_b),
            np.array(state_b, dtype=np.float32), np.array(val_b, dtype=np.float32),
            np.array(rew_b, dtype=np.float32), np.array(done_b, dtype=np.float32),
            np.mean(ep_rews), np.mean(ep_loops), n_blowups)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    env = ResidualMARLEnv(**DESYNC, disable_dw=DISABLE_DW)
    N = env.n
    env.reset(seed=SEED)                 # populate the plant state so env.state() is valid
    state_dim = env.state().shape[0]
    obs_dim = env._obs_space.shape[0]
    act_dim = env._act_space.shape[0]         # 10 = delta_lambda(n=4) + delta_wrench(6)
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim).to(DEVICE)
    critic = Critic(state_dim=state_dim).to(DEVICE)
    if WARMSTART:
        ck = torch.load(WARMSTART, map_location=DEVICE, weights_only=False)
        actor.load_state_dict(ck["state_dict"])
        om = ck["obs_mean"].astype(np.float32); os_ = ck["obs_std"].astype(np.float32)
        crit_msg = ""
        if ck.get("critic_state") is not None:               # warm critic too (avoids the value re-learn dip)
            critic.load_state_dict(ck["critic_state"]); crit_msg = " + critic"
        print(f"actor warm-started from {WARMSTART} (+ its obs normalization{crit_msg})")
    else:
        om, os_ = estimate_norm(env, rng)     # random-rollout obs normalization (mixed-scale {obs_dim}-D)
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
         mean_ep_r, mean_loop, n_blowups) = collect(env, actor, critic, STEPS_PER_ITER, rng, om, os_)

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
            e = eval_policy(env, actor, om, os_)
            hist_det_it.append(it); hist_det.append(e["loop"])
            det_str = (f"  DET_R {e['reward']:.3f}  loop {e['loop']:.3f}  load {e['load']:.3f}"
                       f"  vmin {e['vmin']:.3f}  stall% {100 * e['stallfrac']:.1f}"
                       f"  swing {e['swing']:.3f}  coord {e['coord']:.3f}  jerk {e['jerk']:.3f}")
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
