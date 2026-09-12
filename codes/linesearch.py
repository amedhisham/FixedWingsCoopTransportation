"""linesearch.py — is the -0.39 plateau a ROTATION TAX on a real descent direction, or a true optimum?

Method (scale-free, maximally faithful to training):
  1. load the current policy (WARMSTART), eval baseline DET_R.
  2. run K REAL training iterations (parallel collect + full PPO/Adam update, log_std frozen) and record
     both theta_1 (one iter -> d_inst, includes rotation) and theta_K (net drift -> d_net, rotation averaged).
  3. LINE-SEARCH: eval DET_R at theta_0 + alpha * d  for a sweep of alpha, for BOTH directions.

Reading:
  - d_net keeps DROPPING DET_R past alpha=1 and reaches <= -0.383  -> the averaged direction is genuine
    descent; the spiral is just slow -> ROTATION is the tax, Lookahead/negative-momentum will help.
  - d_net BOTTOMS at alpha~1 then rises                            -> direction only locally valid (must
    re-collect each step) = intrinsic rotation / near-optimum; no straight shortcut.
  - d_net descends but d_inst is FLAT                              -> instantaneous gradient is ~tangential
    (pure circling); only the AVERAGE points downhill = the rotation tax, measured directly.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import time
import numpy as np
import torch
import torch.nn as nn
import mappo
from mappo import (DESYNC, DISABLE_DW, CONSIST_W, CONSIST_LAM_W, OVERFIT_END, T_END, OVERFIT,
                   HIDDEN, WARMSTART, DEVICE, LOG_STD_INIT, LOG_STD_MIN, FREEZE_LOG_STD,
                   LR_ACTOR, LR_CRITIC, GAMMA, LAMBDA, CLIP, ENT_COEF, MAX_GRAD, EPOCHS,
                   MINIBATCH_STEPS, SEED, NUM_WORKERS, eval_policy, compute_gae, overfit_set,
                   critic_input)
from residual_marl_env import ResidualMARLEnv
from networks import Actor, Critic

# Batch MUST be supercritical (critB read 300-700k) so each step's direction is SIGNAL not noise --
# a noisy direction makes a null line-search meaningless. On the CLUSTER set LS_STEPS=1400000 (full
# training batch, ~100s/iter on 90 workers); on the laptop a smaller value trades faithfulness for time.
STEPS = int(os.environ.get("LS_STEPS", 1_400_000))
K = int(os.environ.get("LS_K", 3))          # real training iters; d_inst=after iter 1, d_net=after iter K (rot-averaged)
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def set_flat(params, vec):
    i = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p))
        i += n


def train_iter(actor, critic, opt_a, opt_c, collector, om, os_, om_t, os_t, obs_dim, act_dim, N, rng):
    (obs_b, act_b, logp_b, state_b, rew_b, done_b, dwstar_b, dlamstar_b,
     ep_rews, ep_loops, n_blow) = collector.collect(actor, om, os_, STEPS, base_seed=int(rng.integers(1 << 30)))
    with torch.no_grad():
        val_b = critic(torch.tensor(state_b, device=DEVICE)).cpu().numpy().astype(np.float32)
    T = len(rew_b)
    advs = np.zeros((T, N), np.float32)
    rets = np.zeros((T, N), np.float32)
    for d in range(N):
        advs[:, d], rets[:, d] = compute_gae(rew_b[:, d], val_b, done_b, GAMMA, LAMBDA)
    adv = (advs - advs.mean()) / (advs.std() + 1e-8)
    ret_mean = rets.mean(axis=1)
    tt = lambda a: torch.tensor(a, device=DEVICE)
    obs_t, act_t, logp_old = tt(obs_b), tt(act_b), tt(logp_b)
    state_t, adv_t, ret_t = tt(state_b), tt(adv), tt(ret_mean)
    for _ in range(EPOCHS):
        idx = rng.permutation(T)
        for s in range(0, T, MINIBATCH_STEPS):
            mb = idx[s:s + MINIBATCH_STEPS]
            o = (obs_t[mb].reshape(-1, obs_dim) - om_t) / os_t
            a = act_t[mb].reshape(-1, act_dim)
            lp_old = logp_old[mb].reshape(-1)
            A = adv_t[mb].reshape(-1)
            dist = actor.distribution(o)
            lp = dist.log_prob(a).sum(-1)
            ratio = torch.exp(lp - lp_old)
            l_clip = -torch.min(ratio * A, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A).mean()
            ent = dist.entropy().sum(-1).mean()
            loss_a = l_clip - ENT_COEF * ent
            opt_a.zero_grad()
            loss_a.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD)
            opt_a.step()
            with torch.no_grad():
                actor.log_std.clamp_(min=LOG_STD_MIN)
            v = critic(state_t[mb])
            loss_c = ((v - ret_t[mb]) ** 2).mean()
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD)
            opt_c.step()
    return float(np.mean(ep_rews)), float(np.mean(ep_loops))


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    end_time = OVERFIT_END if OVERFIT else T_END
    env = ResidualMARLEnv(**DESYNC, disable_dw=DISABLE_DW, end_time=end_time,
                          track_clean_lambda=(CONSIST_W > 0 or CONSIST_LAM_W > 0))
    env.reset(seed=SEED)
    env.default_expert_pos = env.expert_pos.copy()
    pairs, eval_scen = overfit_set()
    state_dim = critic_input(env).shape[0]
    obs_dim = env._obs_space.shape[0]
    act_dim = env._act_space.shape[0]
    N = env.n
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=HIDDEN).to(DEVICE)
    critic = Critic(state_dim=state_dim, hidden=HIDDEN).to(DEVICE)
    ck = torch.load(WARMSTART, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ck["state_dict"])
    om = ck["obs_mean"].astype(np.float32)
    os_ = ck["obs_std"].astype(np.float32)
    if ck.get("critic_state") is not None:
        try:
            critic.load_state_dict(ck["critic_state"])
        except RuntimeError:
            print("critic REINIT (state_dim changed)")
    om_t = torch.tensor(om, device=DEVICE)
    os_t = torch.tensor(os_, device=DEVICE)
    actor.log_std.data.fill_(LOG_STD_INIT)
    if FREEZE_LOG_STD:
        actor.log_std.data.fill_(LOG_STD_MIN)
        actor.log_std.requires_grad_(False)
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    base = eval_policy(env, actor, om, os_, eval_scen)
    print(f"baseline DET_R {base['reward']:+.4f}  loop {base['loop']:.4f}  per {base['per_traj_loop']}", flush=True)
    theta0 = flat(actor.parameters()).clone()

    from parallel_collect import ParallelCollector
    env_kwargs = dict(**DESYNC, disable_dw=DISABLE_DW, end_time=end_time,
                      track_clean_lambda=(CONSIST_W > 0 or CONSIST_LAM_W > 0))
    dpos_list = [dpos for _, dpos in pairs]
    collector = ParallelCollector(NUM_WORKERS, env_kwargs, obs_dim, act_dim, HIDDEN, dpos_list)
    torch.set_num_threads(max(1, NUM_WORKERS))
    print(f"running {K} real training iters (STEPS={STEPS}, {NUM_WORKERS} workers) to build the direction...", flush=True)

    theta1 = None
    for it in range(1, K + 1):
        t = time.perf_counter()
        mep, mlp = train_iter(actor, critic, opt_a, opt_c, collector, om, os_, om_t, os_t,
                              obs_dim, act_dim, N, rng)
        if it == 1:
            theta1 = flat(actor.parameters()).clone()
        cur = eval_policy(env, actor, om, os_, eval_scen)
        print(f"  iter {it}: DET_R {cur['reward']:+.4f}  loop {cur['loop']:.4f}  "
              f"(sampled_loop {mlp:.3f})  {time.perf_counter()-t:.0f}s", flush=True)
    thetaK = flat(actor.parameters()).clone()
    d_inst = theta1 - theta0
    d_net = thetaK - theta0
    print(f"\n||d_inst|| (1 iter) = {d_inst.norm():.4e}   ||d_net|| ({K} iters) = {d_net.norm():.4e}", flush=True)

    for name, d in (("d_net  (rotation-averaged, K iters)", d_net), ("d_inst (one iter, has rotation)", d_inst)):
        print(f"\n=== line-search along {name} ===")
        print("alpha   DET_R      loop     blow")
        for al in ALPHAS:
            set_flat(actor.parameters(), theta0 + al * d)
            e = eval_policy(env, actor, om, os_, eval_scen)
            tag = "  <- = the real training move" if al == 1.0 else ""
            print(f"{al:5.1f}  {e['reward']:+.4f}   {e['loop']:.4f}   {e['blowups']}{tag}", flush=True)
    set_flat(actor.parameters(), theta0)
    print("\ndone; actor restored to baseline.", flush=True)


if __name__ == "__main__":
    main()
