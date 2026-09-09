"""
parallel_collect.py — multiprocess rollout collection for MAPPO.

The FMU plant step is the training bottleneck and is inherently SERIAL per env instance. This farms
whole-episode rollouts across a persistent Pool of worker processes, each carrying its OWN FMU env +
trajectory set + a copy of the actor. Each iteration:
  main  -> broadcast the current actor weights + obs-norm + a per-worker seed
  worker-> roll ~STEPS_PER_ITER / NUM_WORKERS steps via mappo.collect_chunk (actor-only, no critic)
  main  -> concatenate the per-worker buffers, then compute the critic value batched (main-side)

Design notes:
  - Workers need ONLY the actor (value is computed in main from state_b), so we ship a small state_dict.
  - Trajectory callables (make_quintic_pose closures) are NOT picklable -> each worker rebuilds its own
    (traj, expert_dpos) set via mappo.build_pairs() at init (same config -> same set). No callables cross
    the process boundary.
  - 'spawn' context (Windows-safe; also fine on Linux/HPC). mappo is imported lazily INSIDE worker fns so
    re-import under spawn can't trigger training (main() stays guarded by __main__).
  - torch is pinned to 1 thread per worker so N processes don't oversubscribe the cores.

Correctness: identical math to the sequential path (same collect_chunk, same batched value). Only the
desync/traj SAMPLING differs by seed across workers — which is the intended domain randomization anyway.
"""
import os
import numpy as np
import torch

_G = {}   # per-worker-process globals (persist across Pool tasks)


def _init(env_kwargs, obs_dim, act_dim, hidden, dpos_list):
    """Runs ONCE per worker process: build the FMU env, the trajectory set, and an actor shell.
    dpos_list = the expert dpos arrays computed ONCE in main -> workers rebuild only the cheap traj
    closures and reuse these, skipping a redundant per-worker CasADi rollout."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # workers IGNORE Ctrl-C; main alone handles it and
    #                                                terminates the pool (else every worker throws
    #                                                KeyboardInterrupt and the join() cleanup hangs).
    import mappo
    from residual_marl_env import ResidualMARLEnv
    from networks import Actor
    torch.set_num_threads(1)                       # one thread/worker -> no oversubscription across processes
    env = ResidualMARLEnv(**env_kwargs)
    env.reset(seed=1000 + (os.getpid() % 30000))
    env.default_expert_pos = env.expert_pos.copy()  # non-TRAJ_RANDOMIZE fallback ref (matches main)
    pairs, n_anchor = mappo.build_pairs(dpos_list)  # cheap closures + shipped dpos -> no per-worker CasADi
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
    actor.eval()
    _G.update(env=env, pairs=pairs, n_anchor=n_anchor, actor=actor)


def _work(args):
    """One collection chunk on this worker's persistent env with freshly-loaded actor weights."""
    import mappo
    state_dict, om, os_, n_steps, seed = args
    _G["actor"].load_state_dict(state_dict)
    rng = np.random.default_rng(seed)
    return mappo.collect_chunk(_G["env"], _G["actor"], om, os_, _G["pairs"], _G["n_anchor"], n_steps, rng)


def _merge(results):
    """Concatenate per-worker buffers (time axis) + pool the per-episode stats."""
    cat = lambda i: np.concatenate([r[i] for r in results], axis=0)
    obs_b, act_b, logp_b, state_b, rew_b, done_b, dwstar_b, dlamstar_b = (cat(i) for i in range(8))
    ep_rews = [x for r in results for x in r[8]]
    ep_loops = [x for r in results for x in r[9]]
    n_blowups = sum(r[10] for r in results)
    return (obs_b, act_b, logp_b, state_b, rew_b, done_b, dwstar_b, dlamstar_b,
            ep_rews, ep_loops, n_blowups)


class ParallelCollector:
    """Persistent Pool of rollout workers. Same output tuple as mappo.collect_chunk (minus value)."""

    def __init__(self, num_workers, env_kwargs, obs_dim, act_dim, hidden, dpos_list):
        import multiprocessing as mp
        self.n = int(num_workers)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.n, initializer=_init,
                             initargs=(env_kwargs, obs_dim, act_dim, tuple(hidden), dpos_list))

    def collect(self, actor, om, os_, total_steps, base_seed):
        sd = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        per = total_steps // self.n
        tasks = [(sd, om, os_, per, int(base_seed) + 7919 * i) for i in range(self.n)]
        results = self.pool.map(_work, tasks)          # blocks until all workers finish their chunk
        return _merge(results)

    def close(self):
        self.pool.close()
        self.pool.join()

    def terminate(self):
        """Force-kill workers (SIGTERM) WITHOUT waiting — the safe path on Ctrl-C / error, since a hung
        or mid-import worker would make close()'s join() block forever."""
        self.pool.terminate()
        self.pool.join()
