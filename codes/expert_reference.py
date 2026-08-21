"""
expert_reference.py — precompute the CLEAN central expert's per-drone loiter path(s).

Runs the central ClassicalAgent (CasADi optimizer) on the NOISE-FREE plant and records every
drone's position/velocity over a trajectory. This path is the IDEAL the F2 RL reward tracks
(r_i penalizes distance to it) and the one demo_desync overlays as "ideal behaviour".

Three entry points:
  expert_path(traj, t_end) -> ONE trajectory's (dpos, dvel, load). Used LIVE by demo_desync for
                              the ideal overlay (one cheap CasADi rollout, always matches the traj).
  build_library(out)       -> the per-trajectory DICTIONARY (ragged) over the RL-training set (the
                              SAME 56 trajs collect_prdot uses) + the showcase demo sets, keyed in
                              expert_lib.npz. The multi-traj RL reward reads this per episode.
  main()                   -> back-compat single default-trajectory expert_ref.npz.

Regenerate whenever get_reference_trajectory / the trajectory library changes.
Run:  python expert_reference.py       (default expert_ref.npz)
      python expert_reference.py lib   (full expert_lib.npz)
"""

import numpy as np
from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from trajectories import train_set, custom_set, showcase_set
from collect_prdot_data import N_TRAJ

OUT = "expert_ref.npz"
LIB_OUT = "expert_lib.npz"


def _run_expert(env, agent, traj, t_end):
    """Roll the noise-free central ClassicalAgent on `traj` for t_end s (env+agent REUSED, reset
    here). Returns dpos (N,T,3), dvel (N,T,3), load (T,3)."""
    obs, _ = env.reset()
    agent.reset()
    prev_f = np.array([0.0, 0.0, FZ] * N)
    dpos = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    load = []
    t = 0.0
    while t < t_end - 1e-9:
        pos = obs[0:3]
        R = np.round(obs[3:12].reshape((3, 3), order="C"), 6)
        vel, w = obs[12:15], obs[15:18]
        f, _, _ = agent.compute_forces(pos, vel, R, w, t, traj=traj)   # traj-aware central expert
        ff = LLC_ALPHA * f + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        for i in range(N):
            dpos[i].append(obs[18 + 3 * i: 18 + 3 * i + 3].copy())
            dvel[i].append(obs[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3].copy())
        load.append(pos.copy())
        obs, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    return np.array(dpos), np.array(dvel), np.array(load)


def expert_path(traj, t_end=T_END):
    """Convenience: throwaway env+agent, run the expert once on `traj` -> (dpos, dvel, load).
    Used LIVE by demo_desync for the ideal overlay."""
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=max(t_end, T_END))
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)
    dpos, dvel, load = _run_expert(env, agent, traj, t_end)
    env.close()
    return dpos, dvel, load


def _library_entries():
    """(setname, label, traj, t_end) for every expert ref to precompute.
    RL-training pool = the SAME set collect_prdot uses (default + customs + N_TRAJ quintics, all at
    T_END); demo pool = showcase short + long (held out from training, their own t_ends)."""
    ents = [("train", "default", None, T_END)]
    for i, (tr, _p) in enumerate(custom_set()):
        ents.append(("train", f"custom_{i}", tr, T_END))
    for i, (tr, _p) in enumerate(train_set(N_TRAJ)):
        ents.append(("train", f"quintic_{i}", tr, T_END))
    for kind in ("short", "long"):
        for label, tr, t_end in showcase_set(kind):
            ents.append((f"showcase_{kind}", label, tr, t_end))
    return ents


def build_library(out=LIB_OUT):
    """Precompute the per-trajectory expert DICTIONARY (ragged) -> `out`. Each entry keyed
    `{set}__{idx}` stores dpos/dvel/load; a manifest (meta_set/label/tend/key) drives reload.
    One reused env+agent. WARNING: 56+ CasADi rollouts -> minutes; offline one-time."""
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)   # T_END is the max horizon (35)
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    ents = _library_entries()
    data, meta_set, meta_label, meta_tend, meta_key = {}, [], [], [], []
    per_set_idx = {}
    for setname, label, traj, t_end in ents:
        idx = per_set_idx.get(setname, 0)
        per_set_idx[setname] = idx + 1
        key = f"{setname}__{idx}"
        dpos, dvel, load = _run_expert(env, agent, traj, t_end)
        data[f"{key}__dpos"] = dpos
        data[f"{key}__dvel"] = dvel
        data[f"{key}__load"] = load
        meta_set.append(setname); meta_label.append(label)
        meta_tend.append(float(t_end)); meta_key.append(key)
        print(f"  {key:16s} {label:14s} t_end {t_end:>4}  dpos {dpos.shape}")
    env.close()

    np.savez(out, meta_set=np.array(meta_set), meta_label=np.array(meta_label),
             meta_tend=np.array(meta_tend, float), meta_key=np.array(meta_key), **data)
    print(f"\nsaved {out}: {len(ents)} expert refs  "
          f"({', '.join(f'{k}={v}' for k, v in per_set_idx.items())})")


def load_library(path=LIB_OUT):
    """Reload -> {setname: [ {label, t_end, dpos, dvel, load} ordered by idx ]}."""
    z = np.load(path, allow_pickle=False)
    lib = {}
    for setname, label, t_end, key in zip(z["meta_set"], z["meta_label"], z["meta_tend"], z["meta_key"]):
        lib.setdefault(str(setname), []).append(dict(
            label=str(label), t_end=float(t_end),
            dpos=z[f"{key}__dpos"], dvel=z[f"{key}__dvel"], load=z[f"{key}__load"]))
    return lib


def training_pairs(lib_path=LIB_OUT):
    """[(traj, expert_dpos)] for the RL-training 'train' set, INDEX-ALIGNED: the traj callables are
    rebuilt from the SAME generators _library_entries() used (default + custom_set + train_set(N_TRAJ)),
    so trajs[i] is exactly the reference the lib's train[i] expert path was computed on. Used by mappo
    to sample a (reference, expert) pair per episode. traj=None means the default trajectory.
    REGENERATE the lib (python expert_reference.py lib) whenever that set / N_TRAJ changes."""
    train = load_library(lib_path)["train"]
    anchors = [None] + [tr for tr, _ in custom_set()]        # NON-quintic: default line + solver-engaging customs
    trajs = anchors + [tr for tr, _ in train_set(N_TRAJ)]
    assert len(trajs) == len(train), (
        f"traj/lib misalignment: {len(trajs)} trajs vs {len(train)} refs — rebuild expert_lib.npz")
    pairs = [(trajs[i], train[i]["dpos"]) for i in range(len(trajs))]
    n_anchor = len(anchors)             # pairs[:n_anchor] are the non-quintic anchors (indices 0..n_anchor-1)
    return pairs, n_anchor


def eval_scenarios(lib_path=LIB_OUT):
    """Fixed 2-traj eval set for best-net SELECTION (score = MEAN over it, not one lucky stick):
    1 straight LINE (the default trajectory, in-distribution anchor) + 1 HELD-OUT QUINTIC (showcase
    long_quintic1, drawn from SHOWCASE_SEED -> disjoint from the 56 training trajs, so it doubles as a
    generalization signal). Returns [(label, traj, expert_dpos)]; traj=None means the default line."""
    lib = load_library(lib_path)
    line = ("line", None, lib["train"][0]["dpos"])           # train[0] is the default == expert_ref.npz
    q_label, q_traj, _t = showcase_set("long")[1]            # long_quintic1 (35 s), HELD OUT
    quintic = (q_label, q_traj, lib["showcase_long"][1]["dpos"])
    return [line, quintic]


def main():
    """Back-compat: single default-trajectory expert path -> expert_ref.npz."""
    dpos, dvel, load = expert_path(None, T_END)
    np.savez(OUT, dpos=dpos, dvel=dvel, load=load)
    print(f"saved {OUT}  dpos {dpos.shape}  (per-drone loiter path over {dpos.shape[1]} steps)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "lib":
        build_library()
    else:
        main()
