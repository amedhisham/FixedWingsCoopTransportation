"""
dagger_prdot.py — DAgger for the pR_dot whole-vector lambda policy.

The BC policy (il_actor_prdot.pt) fit the optimizer perfectly (R^2=1.0) but buzzes
in closed loop: it reconstructs pR_dot from its OWN lambda_{t-1}, which drifts off
the manifold the OPTIMIZER's lambda produced during collection. Classic covariate
shift — the self-fed input distribution was never trained on.

DAgger fixes it: roll out the CURRENT policy, but build its input (pR_dot + lambda_{t-1})
from the ACTUALLY-APPLIED lambda history (the state it really visits), label every
visited state with the EXPERT (optimizer) lambda, aggregate, retrain, repeat. As
beta -> 0 the applied lambda -> the policy's own, so the training distribution
converges to the deployment distribution.

Two enablers, same as F1:
  - The clock keeps the off-manifold labels a consistent function of the input.
  - G.N = 0 makes beta-MIXING safe: applied lambda = beta*expert + (1-beta)*policy
    only touches the nullspace, so any blend leaves load tracking intact.

Whole-vector: ONE 30-D input row -> one lambda[N] label per step (not 4 per-drone rows).
Saves il_actor_prdot_dagger.pt, prints per-iter retrain MSE + buzz, then runs
deploy_prdot on the final policy.
"""

import copy
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from optimizer import cable_force_calculation
from controller import error_calculation, get_reference_trajectory
from networks import Actor
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from collect_prdot_data import (Reconstructor, build_input, LAM0, SUFFIX,
                                 PRDOT_HIDDEN, activity_bin, LAM_LP_TAU,
                                 init_lam_history, push_lam)
from controller import get_reference_trajectory
from deploy_prdot import main as deploy_prdot_main
from trajectories import train_set, custom_set, TRAIN_SEED
import os

BYPASS_OPT = False   # adaptive optimizer (matches prdot_dataset.npz)
AGG_OUT = f"prdot_dagger_aggregate{SUFFIX}.npz"   # persisted DAgger aggregate -> real resume
TRAJ_PER_ITER = 30   # GENERALIZATION: closed-loop rollouts per DAgger iter, each on a fresh
                     #   sampled quintic-pose trajectory (was 10). More coverage per iter.
INCLUDE_DEFAULT = True   # also roll out the ORIGINAL straight-line trajectory each iter (anchor),
                         #   matching collect_prdot_data's include_default.
MAX_AGG = 250_000    # cap the persisted POOL of OLD data (what we sample from). ~2 iters of new data
                     #   (126k/iter at T_END=35). Eviction: hardest-first (USE_CURATION) else uniform.
TRAIN_BUDGET = 200_000   # per-iter TRAINING set size. We train on ALL of this iter's new data FULLY,
                     #   then TOP UP with a sample of the OLD pool to reach this budget (new is NEVER
                     #   masked away). Was: concat old+new, evict, train on the WHOLE 150k -> the
                     #   "very long training + new data could be evicted" disaster.
WARM_START = True    # retrain from the PREVIOUS net (not fresh) each iter -> fewer epochs suffice.
USE_CURATION = False  # False = UNIFORM (random old-sample + random eviction). True = HARDEST-first,
                     #   deterministic top-K (interim; the bin-quota / separate-filters rebuild is
                     #   pending — see memory f1-curate-redesign). Either way: new data trained FULLY.

# beta schedule: 1 = pure expert (stay on the optimizer manifold), 0 = pure policy
# (deployment distribution). CONTINUE mode: all pure-policy iters to collect+correct the
# closed-loop buzz. Each entry = one DAgger iter (x (1 default + TRAJ_PER_ITER) rollouts).
BETAS = [0.0,0.0] #0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0.08,0.06,0.04,0.02,0.0

# ADAPTIVE beta ladder near deployment: don't leave a beta until the closed-loop buzz drops below
# BUZZ_PASS. Warm-start transmits a jittery net's bad weight-basin (hard-won: a regressed net is
# sticky, MSE is only a strong-not-perfect buzz proxy), so we gate on buzz directly and retry the
# SAME beta rather than shoving a not-yet-good net down to a harder one. Only matters near beta=0.
GATE_BETA = 0.1       # gating active only for beta <= this (deployment region); above -> one pass/beta
BUZZ_PASS = 0.022     # leave a gated beta once mean rollout buzz < this (floor is ~0.016)
STUCK_AFTER = 0       # normal-budget retries before escalating to keep-all-data + train-whole-pool.
                      #   0 = the FIRST failed gate is already "stuck" -> train on EVERYTHING at once
                      #   (a buzz problem needs the hard data now). Gating alone (beta<=GATE_BETA)
                      #   still does NOT bloat: a beta that PASSES stays at the normal caps.
MAX_RETRIES = 3       # total retrains at a stuck gated beta before advancing anyway (best-effort)
EPOCHS = 150          # fewer epochs: warm-started from the previous net each iter (WARM_START)
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.2
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-iteration plots auto-close after this many seconds so the run proceeds
# unattended. Set 0 to block on every iteration. Final buzz curve + deploy block.
PAUSE_SEC = 20.0


def rollout(policy, om, os_, beta, env, agent, Bb, L0, traj=None):
    """Drive the plant with lambda = beta*expert + (1-beta)*policy for one episode on
    reference `traj` (None -> default). env + agent are REUSED across rollouts (reset here).

    The policy input (pR_dot + lambda_{t-1}) is reconstructed from the APPLIED
    (mixed) lambda history — the state distribution the policy actually visits.
    Returns visited-state inputs (M,30) labelled with EXPERT lambda (M,N) + diag.
    """
    obs42, _ = env.reset()
    agent.reset()

    prev_f = np.array([0.0, 0.0, FZ] * N)
    prev_lam = LAM0.copy()          # APPLIED lambda_{t-1} (post-EMA); for the pR_dot reconstruction
    lam_buf = init_lam_history()    # newest-first last-LAM_HIST applied lambda (net input)
    prev_vLd = None                 # ref velocity_{t-1} (accel -> activity bin)
    lam_lp = LAM0.copy()            # in-loop EMA state (soft warm-start continuity)
    lam_a = None if LAM_LP_TAU is None else DT / (LAM_LP_TAU + DT)
    recon = Reconstructor(Bb, L0, DT)

    X_rows, Y_rows, H_rows, bin_rows = [], [], [], []
    lam_pol_hist = [[] for _ in range(N)]
    lam_exp_hist = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    load_hist, ref_hist = [], []

    t = 0.0
    while t < T_END - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        # w_d ONCE (advances integrators once); reused for input, expert, and drive.
        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t, traj)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)

        # Policy input at the VISITED state: pR_dot (mode set by ANALYTIC) from lambda_{t-1}.
        vR = recon(R, vel, angvel, w_d, prev_lam)
        row = build_input(t, vR, lam_buf)
        Xn = ((row[None, :] - om) / os_).astype(np.float32)
        with torch.no_grad():
            lam_pol = policy(torch.tensor(Xn)).numpy().flatten()

        # Expert label at the SAME visited state = the DAgger target.
        lam_exp, _ = agent.optimize(t, R, vel, angvel, w_d, bypass=BYPASS_OPT)

        # Rollout driven by the MIX; label is the pure EXPERT.
        lam_mixed = beta * lam_exp + (1.0 - beta) * lam_pol
        # IN-LOOP EMA: smooth the APPLIED lambda (soft warm-start continuity). The smoothed
        # lambda drives the plant AND feeds back (recon/prev_lam) -> the net trains against the
        # SAME filtered feedback it sees at deploy (deploy_prdot applies the identical EMA).
        if lam_a is not None:
            lam_lp = lam_a * lam_mixed + (1.0 - lam_a) * lam_lp
            lam_applied = lam_lp
        else:
            lam_applied = lam_mixed
        f_full, _ = cable_force_calculation(R, Bb, w_d, lam_applied, N)

        # Hardness = DAgger disagreement ||expert - policy|| (the informative samples);
        # activity bin from the reference kinematics (speed + finite-diff accel).
        ref = get_reference_trajectory(t, traj)
        v_Ld = ref[1]
        accel = 0.0 if prev_vLd is None else float(np.linalg.norm(v_Ld - prev_vLd) / DT)
        prev_vLd = v_Ld.copy()

        X_rows.append(row)
        Y_rows.append(lam_exp)
        H_rows.append(float(np.linalg.norm(lam_exp - lam_pol)))
        bin_rows.append(activity_bin(float(np.linalg.norm(v_Ld)), accel))
        for i in range(N):
            lam_pol_hist[i].append(lam_pol[i])
            lam_exp_hist[i].append(lam_exp[i])
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        load_hist.append(pos.copy())
        ref_hist.append(ref[0].copy())

        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))

        # Roll histories with the APPLIED (EMA-smoothed mixed) lambda.
        recon.roll(lam_applied)
        prev_lam = lam_applied.copy()
        lam_buf = push_lam(lam_buf, lam_applied)
        t += DT

    lam_pol_hist = [np.array(l) for l in lam_pol_hist]
    lam_exp_hist = [np.array(l) for l in lam_exp_hist]
    dvel = [np.array(v) for v in dvel]
    load = np.array(load_hist); ref = np.array(ref_hist)
    buzz = float(np.mean([np.mean(np.abs(np.diff(l))) for l in lam_pol_hist]))
    diag = {
        "buzz": buzz,
        "vmin": float(min(v.min() for v in dvel)),
        "vmean": float(np.mean([v.mean() for v in dvel])),
        "track_mean": float(np.linalg.norm(load - ref, axis=1).mean()),
        "track_max": float(np.linalg.norm(load - ref, axis=1).max()),
        "lam_pol": lam_pol_hist,
        "lam_exp": lam_exp_hist,
        "dvel": dvel,
        "t": np.arange(len(load)) * DT,
    }
    return (np.asarray(X_rows, dtype=np.float32),
            np.asarray(Y_rows, dtype=np.float32),
            np.asarray(H_rows, dtype=np.float32),
            np.asarray(bin_rows, dtype=np.int64),
            diag)


def train(X, Y, init_state=None):
    """Retrain on the aggregated set. Fresh normalization, 80/20 split, best-val checkpoint.
    WARM_START: init_state (previous net's cpu state_dict) seeds the weights so fewer epochs
    suffice. Returns cpu state_dict + stats + best-val MSE + Var(lambda)."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    xm = X.mean(0, keepdims=True)
    xs = X.std(0, keepdims=True) + 1e-6
    Xn = ((X - xm) / xs).astype(np.float32)

    M = len(X)
    idx = rng.permutation(M)
    n_val = int(VAL_FRAC * M)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tt = lambda a: torch.tensor(a, device=DEVICE)
    tr_x, tr_y = tt(Xn[tr_idx]), tt(Y[tr_idx])
    va_x, va_y = tt(Xn[val_idx]), tt(Y[val_idx])
    loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=BATCH, shuffle=True)

    net = Actor(obs_dim=X.shape[1], act_dim=Y.shape[1], hidden=PRDOT_HIDDEN).to(DEVICE)
    if init_state is not None:                       # WARM_START: seed from the previous net
        net.load_state_dict(init_state)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    best_va, best_state = float("inf"), None
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); mse(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = mse(net(va_x), va_y).item()
        if va < best_va:
            best_va = va
            best_state = copy.deepcopy({k: v.cpu() for k, v in net.state_dict().items()})
    return best_state, xm.astype(np.float32), xs.astype(np.float32), best_va, float(Y.var())


def show_diag(diag, label):
    """Overlay expert-vs-policy lambda + velocity norms for one rollout."""
    fig1, ax1 = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(ax1):
        ax.plot(diag["t"], diag["lam_exp"][i], "k", lw=1.0, label="expert (would-do)")
        ax.plot(diag["t"], diag["lam_pol"][i], "m", lw=1.0, alpha=0.8, label="policy")
        ax.set_ylabel(f"$\\lambda_{i+1}$"); ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    ax1[-1].set_xlabel("Time (s)"); fig1.suptitle(f"Expert vs policy lambda — {label}")

    plt.figure()
    for i in range(N):
        plt.plot(diag["t"], diag["dvel"][i], label=f"Drone {i+1}")
    plt.axhline(EPS, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms — {label}"); plt.legend(); plt.grid(True)

    if PAUSE_SEC > 0:
        plt.show(block=False)
        plt.pause(PAUSE_SEC)
        plt.close("all")
    else:
        plt.show()


def main():
    # Warm-start (RESUME-aware): prefer the DAGGERED net over BC, and the saved AGGREGATE over
    # the collect dataset -> repeated runs truly CONTINUE (build on prior corrections + the
    # improved policy's state distribution), instead of restarting from BC + collect each time.
    dagger_ckpt = f"il_actor_prdot_dagger_autosave_prev_analytic.pt"
    net_path = dagger_ckpt if os.path.exists(dagger_ckpt) else f"il_actor_prdot{SUFFIX}.pt"
    ckpt = torch.load(net_path, map_location="cpu", weights_only=False)
    hidden = tuple(ckpt.get("hidden", (128, 128)))     # pre-"hidden" ckpts were (128,128)
    policy = Actor(obs_dim=ckpt["obs_mean"].shape[1], act_dim=N, hidden=hidden)
    policy.load_state_dict(ckpt["state_dict"]); policy.eval()
    om, os_ = ckpt["obs_mean"].astype(np.float32), ckpt["obs_std"].astype(np.float32)

    data_path = AGG_OUT if os.path.exists(AGG_OUT) else f"prdot_dataset{SUFFIX}.npz"
    data = np.load(data_path)
    D_X = data["X"].astype(np.float32)
    D_Y = data["Y"].astype(np.float32)
    # H/bins for hardness curation. Old npz without them -> keep-all fallback (H=1, bin=cruise).
    D_H = data["H"].astype(np.float32) if "H" in data else np.ones(len(D_X), np.float32)
    D_bin = data["bins"].astype(np.int64) if "bins" in data else np.ones(len(D_X), np.int64)
    # Age (DAgger generation) for recency bias. Loaded data = generation 0; new iters increment.
    D_age = data["age"].astype(np.int64) if "age" in data else np.zeros(len(D_X), np.int64)
    gen0 = int(D_age.max()) + 1 if len(D_age) else 0
    print(f"warm-start net <- {net_path}  hidden={hidden}\n"
          f"dataset        <- {data_path}  ({len(D_X)} samples)\n")

    # Reuse ONE env + ONE agent (expensive CasADi solver) across all rollouts.
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)
    curate_rng = np.random.default_rng(SEED)

    def snapshot():                     # deep-copy the CURRENT policy weights (late-binds `policy`)
        return {k: v.clone() for k, v in policy.state_dict().items()}
    good_state = snapshot()             # last VERIFIED-GOOD net (warm-start source + revert target)

    # Ctrl-C autosave: keep the last 2 CLEAN (verified-good) nets only — never a stuck/retry
    # candidate. keep_good() is called wherever good_state is anchored, with the om/os_ that
    # MATCH that net at that moment (so the saved triple deploys correctly).
    recent = deque(maxlen=2)
    def keep_good():
        recent.append({"state_dict": {k: v.clone() for k, v in good_state.items()},
                       "obs_mean": np.asarray(om).copy(), "obs_std": np.asarray(os_).copy(),
                       "hidden": hidden})
    def flush_autosave():
        if not recent:
            print("\n[Ctrl-C] no verified-good net yet — nothing to autosave"); return
        tags = ["_prev", "_last"][-len(recent):]   # 1 item -> "_last"; 2 -> ["_prev","_last"]
        for ck, tag in zip(recent, tags):
            p = f"il_actor_prdot_dagger_autosave{tag}{SUFFIX}.pt"
            torch.save(ck, p); print(f"[Ctrl-C] autosaved {p}")
    keep_good()                         # the warm-start net is itself a clean starting point

    def evict_pool():                   # collapse the pool back to the normal MAX_AGG cap
        nonlocal D_X, D_Y, D_H, D_bin, D_age
        if len(D_X) > MAX_AGG:
            keep = np.argsort(-D_H)[:MAX_AGG] if USE_CURATION else curate_rng.permutation(len(D_X))[:MAX_AGG]
            D_X, D_Y, D_H, D_bin, D_age = D_X[keep], D_Y[keep], D_H[keep], D_bin[keep], D_age[keep]

    def add_new():                      # append THIS pass's new rollout data to the pool
        nonlocal D_X, D_Y, D_H, D_bin, D_age
        D_X = np.concatenate([D_X, nX]); D_Y = np.concatenate([D_Y, nY])
        D_H = np.concatenate([D_H, nH]); D_bin = np.concatenate([D_bin, nbin])
        D_age = np.concatenate([D_age, np.full(len(nX), gen0 + it - 1, np.int64)])

    def budgeted_trainset():            # NORMAL train set: ALL new (fully) + sample old to TRAIN_BUDGET
        n_old = max(0, TRAIN_BUDGET - len(nX))    # call BEFORE add_new (samples OLD pool, excludes new)
        if len(D_X) <= n_old:
            old_idx = np.arange(len(D_X))
        elif USE_CURATION:
            old_idx = np.argsort(-D_H)[:n_old]                     # HARDEST-first, deterministic
        else:
            old_idx = curate_rng.choice(len(D_X), n_old, replace=False)   # uniform
        trX = np.concatenate([nX, D_X[old_idx]]) if len(D_X) else nX
        trY = np.concatenate([nY, D_Y[old_idx]]) if len(D_X) else nY
        return trX, trY

    buzz_curve = []
    bi, it, retries = 0, 0, 0
    interrupted = False
    while bi < len(BETAS):
      try:
        beta = BETAS[bi]; it += 1
        gate_on = beta <= GATE_BETA

        # ---- rollout the CURRENT policy at beta -> new data + closed-loop buzz ----
        # Seed off the GLOBAL generation (gen0 + it - 1), so staged runs + retries keep drawing
        # FRESH trajectories from the stream instead of re-rolling the same set.
        batch = list(train_set(TRAJ_PER_ITER, seed=TRAIN_SEED + gen0 + it - 1))
        batch = custom_set() + batch           # 5 solver-engaging customs every iter (fixed set)
        if INCLUDE_DEFAULT:
            batch.insert(0, (None, None))      # anchor: closed-loop correction on the straight line too
        nX, nY, nH, nbin = [], [], [], []
        buzz_k, vmin_k, track_k, def_diag = [], [], [], None
        for traj, _p in batch:
            new_X, new_Y, new_H, new_bin, diag = rollout(policy, om, os_, beta, env, agent, Bb, L0, traj)
            nX.append(new_X); nY.append(new_Y); nH.append(new_H); nbin.append(new_bin)
            buzz_k.append(diag["buzz"]); vmin_k.append(diag["vmin"]); track_k.append(diag["track_mean"])
            if traj is None:
                def_diag = diag                      # the straight-line anchor: the cross-iter reference
        nX = np.concatenate(nX); nY = np.concatenate(nY)
        nH = np.concatenate(nH); nbin = np.concatenate(nbin)
        buzz = float(np.mean(buzz_k))
        if def_diag is not None:
            show_diag(def_diag, f"iter {it} default-anchor  (beta {beta:.2f}, retry {retries})")
        buzz_curve.append(buzz)

        # ================= GATED region (beta <= GATE_BETA): STAY until buzz passes ==============
        if gate_on:
            passed = buzz < BUZZ_PASS
            giveup = (not passed) and retries >= MAX_RETRIES

            if passed or giveup:
                # ADVANCE — but ALWAYS train on this pass's fresh data first (never waste a rollout).
                # On PASS, anchor good_state to the VERIFIED (pre-train) net; carry the freshly-trained
                # net forward (the next beta's gate re-verifies it). Normal budget (advancing != stuck).
                if passed:
                    good_state = snapshot()           # verified passing net = warm-start/revert anchor
                    keep_good()                        # clean net (om/os_ still match it here) -> autosave buffer
                init = policy.state_dict() if passed else good_state
                trX, trY = budgeted_trainset()
                state, om, os_, best_va, var_lam = train(trX, trY, init_state=init)
                policy = Actor(obs_dim=trX.shape[1], act_dim=N, hidden=hidden)
                policy.load_state_dict(state); policy.eval()
                add_new(); evict_pool()               # keep data, pool back to the normal MAX_AGG cap
                tag = "PASS" if passed else f"GIVE-UP/{MAX_RETRIES}"
                print(f"iter {it}  beta {beta:.2f}  {tag}  buzz {buzz:.4f} "
                      f"{'<' if passed else '>='} {BUZZ_PASS}  |  MSE {best_va:.4f}  "
                      f"train {len(trX)}  pool {len(D_X)}  -> advance")
                bi += 1; retries = 0
                continue

            # RETRY (stay at this beta): warm-start from the last VERIFIED-GOOD net (never a candidate).
            retries += 1
            stuck = retries > STUCK_AFTER             # ACTUALLY stuck -> escalate to the whole pool
            if stuck:
                add_new()                             # keep ALL data (no evict), train on the whole pool
                trX, trY = D_X, D_Y
            else:
                trX, trY = budgeted_trainset()        # NORMAL: new-fully + sampled-old to TRAIN_BUDGET
                add_new(); evict_pool()               #   pool stays at the normal MAX_AGG cap
            state, om, os_, best_va, var_lam = train(trX, trY, init_state=good_state)
            policy = Actor(obs_dim=trX.shape[1], act_dim=N, hidden=hidden)
            policy.load_state_dict(state); policy.eval()
            print(f"iter {it}  beta {beta:.2f}  retry {retries}/{MAX_RETRIES}"
                  f"{' STUCK' if stuck else ''}  buzz {buzz:.4f} >= {BUZZ_PASS}  |  "
                  f"MSE {best_va:.4f}  train {len(trX)}  pool {len(D_X)}")
            continue                                  # stay at this beta; next loop evaluates the retrain

        # ================= NON-GATED (beta > GATE_BETA): one pass, then advance =================
        # TRAIN SET = ALL new data (FULLY) + a sample of the OLD pool to reach TRAIN_BUDGET.
        trX, trY = budgeted_trainset()

        init = policy.state_dict() if WARM_START else None
        state, om, os_, best_va, var_lam = train(trX, trY, init_state=init)
        policy = Actor(obs_dim=trX.shape[1], act_dim=N, hidden=hidden)
        policy.load_state_dict(state); policy.eval()
        good_state = snapshot()                       # non-gated betas are accepted as-is
        keep_good()                                   # clean net (om/os_ = just-trained) -> autosave buffer

        add_new(); evict_pool()                       # pool += new, bound to MAX_AGG (normal cap)

        print(f"iter {it}  beta {beta:.2f}  ({len(batch)} trajs)  |  buzz {buzz:.4f}  "
              f"vmin {min(vmin_k):.3f}  track {np.mean(track_k):.4f}  |  "
              f"MSE {best_va:.4f} (Var {var_lam:.3f})  |  new {len(nX)} train {len(trX)} pool {len(D_X)}")
        bi += 1
      except KeyboardInterrupt:
        interrupted = True
        break                                         # bail out of the ladder, autosave below
    env.close()

    if interrupted:
        print("\n[Ctrl-C] interrupted — autosaving the last 2 verified-good nets")
        flush_autosave()
        return

    torch.save({"state_dict": {k: v for k, v in policy.state_dict().items()},
                "obs_mean": om, "obs_std": os_, "hidden": hidden}, f"il_actor_prdot_dagger{SUFFIX}.pt")
    np.savez(AGG_OUT, X=D_X, Y=D_Y, H=D_H, bins=D_bin, age=D_age)   # persist aggregate -> next run RESUMEs
    print(f"\nsaved il_actor_prdot_dagger{SUFFIX}.pt  +  {AGG_OUT}  ({len(D_X)} samples)")

    plt.figure()
    plt.plot(range(1, len(buzz_curve) + 1), buzz_curve, "o-")
    plt.xlabel("DAgger iteration"); plt.ylabel("policy action buzz  (mean |d lambda/step|)")
    plt.title("Closed-loop lambda buzz across DAgger"); plt.grid(True)
    plt.show()

    print(f"\n--- deploy_prdot on il_actor_prdot_dagger{SUFFIX}.pt ---")
    deploy_prdot_main(f"il_actor_prdot_dagger{SUFFIX}.pt")


if __name__ == "__main__":
    main()
