"""
warmstart_residual.py — STAGE 1 of F2: supervised warm-start for the residual actor.

The frozen analytic pR_dot base is a clean, noise-naive imitator: at zero noise the four
local replicas agree and the load tracks; under desync the per-drone views diverge, the
replicas' forces stop cancelling, and the load drifts. The residual's job is to fix that
drift with a full-authority (Cartesian) force correction the base can't produce (the
base's lambda only touches the nullspace, G.N=0).

Before RL, we give the residual a supervised head start by regressing it toward the
PRIVILEGED-optimal correction — the gap between what a clean centralized optimizer would
command and what the desynced base actually produced:

    target  (per drone i) :  delta_f*_i = f_central_i - f_local_i
      f_local_i  = drone i's OWN base force from its NOISY view (base lambda + its w_d),
                   sliced i  =  f_g_i + f_lambda_i          # the force it really applies
      f_central_i= a CLEAN ClassicalAgent/optimizer on the TRUE (noise-free) state,
                   full force, sliced i                     # what it SHOULD have applied
    obs (30-D, per drone i):  [ noisy load est(18), own drone(6), f_g_i(3), f_lambda_i(3) ]

The target is computable only at training time (privileged / centralized) but the obs is
purely local -> CTDE. Pure MSE (have a target -> imitate; NOT a reward). The LABEL is
delta_f* clipped to the deployable cap (RESIDUAL_CAP*||f_local||) -- the env caps applied
delta_f anyway, so fitting beyond it teaches saturation and lets a few over-cap outliers
dominate the MSE. The DRIVE keeps the UNclipped delta_f* (so beta=1 -> f_central exactly).
delta_f* itself is fixed (base-only f_local, clean f_central); only the state dist moves.
DAgger with a beta schedule -- drive = f_local + [beta*delta_f* + (1-beta)*residual]:
beta=1 drives f_central (the clean centralized trajectory, oracle correction), beta=0
drives the pure residual (deployment). We relabel every visited state with delta_f*, so
as beta -> 0 the net trains on the states it will actually see. Unlike the base DAggers,
the label (delta_f*) and the drive are DIFFERENT objects, so beta mixes the oracle
CORRECTION into the drive rather than mixing an expert action with the policy.

Same class as the RL actor (Actor 30->3): this checkpoint is loaded DIRECTLY as the MAPPO
warm-start in Stage 2. Output = delta_f in Newtons; with residual_scale=1.0 that IS the
env action mean, capped at residual_cap*||f_base|| just like the env. Saves
residual_warmstart.pt (state_dict + obs_mean + obs_std).
"""

import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from controller import error_calculation, get_reference_trajectory
from networks import Actor
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from collect_prdot_data import RECON_ALPHA
from residual_marl_env import LocalModelAgent
# reuse the EXACT desync sensor + spec from noisy DAgger (single source of truth)
from dagger_noisy import DesyncSensor, DELAY_CHOICES, CLOCK_OFFSET

BASE_POLICY = "il_actor_prdot_dagger_analytic.pt"   # frozen smooth base (chosen residual base)
BYPASS_OPT = False                                  # clean adaptive optimizer as f_central source

# Residual action semantics — MUST match ResidualMARLEnv (so the warm-start seeds RL cleanly).
RESIDUAL_SCALE = 1.0        # net output = delta_f in Newtons
RESIDUAL_CAP = 0.5          # applied ||delta_f|| <= RESIDUAL_CAP * ||f_base_i||

# DAgger schedule. Drive = f_local + [beta*delta_f* + (1-beta)*residual]:
#   beta=1 -> f_local + delta_f* = f_central  -> the CLEAN centralized trajectory (oracle
#             correction; collect labels on well-tracked states),
#   beta=0 -> f_local + residual              -> the DEPLOYMENT distribution.
# The LABEL delta_f* is beta-independent (clean f_central - base-only f_local); beta only
# slides WHICH states we visit. Same structure as dagger_noisy, delta_f* replacing lam_clean.
BETAS = [1.0, 1.0]

EPOCHS = 250
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.2
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAUSE_SEC = 20.0


def _cap(df, base_i):
    """Norm-cap a residual to RESIDUAL_CAP*||base_i|| — identical to ResidualMARLEnv.step."""
    max_res = RESIDUAL_CAP * np.linalg.norm(base_i)
    nrm = np.linalg.norm(df)
    if max_res > 0 and nrm > max_res:
        df = df * (max_res / nrm)
    return df


def rollout(base_net, om, os_, resnet, res_om, res_os, beta, rng):
    """One decentralized desync episode. Drive = f_local + [beta*delta_f* + (1-beta)*residual].

    Records the privileged residual target for EVERY visited (drone, step):
        X row  = obs33_i  ,  Y row = delta_f*_i = f_central_i - f_local_i.
    delta_f* is beta-independent; beta only slides the visited-state distribution (beta=1
    -> clean centralized trajectory, beta=0 -> pure-residual deployment). resnet=None ->
    the residual term is 0 (iter 1, before any residual exists). Returns (M,33), (M,3), diag.
    """
    delays = rng.integers(DELAY_CHOICES[0], DELAY_CHOICES[1] + 1, size=N)
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    clean = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)                 # f_central on TRUE state
    locals_ = [LocalModelAgent(N, DT, PHASES, EPS, L0, m, J, Bb, RECON_ALPHA) for _ in range(N)]
    sensor = DesyncSensor(N, delays, rng)

    prev_f = np.array([0.0, 0.0, FZ] * N)
    X_rows, Y_rows = [], []
    dstar = [[] for _ in range(N)]      # ||delta_f*_i|| (target magnitude)
    dvel = [[] for _ in range(N)]
    cap_hit = 0                          # steps where ||delta_f*|| exceeds the deployable cap
    load_hist, ref_hist = [], []

    t = 0.0
    while t < T_END - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        # Per-drone noisy/delayed views -> base net input rows.
        ests = sensor.update(pos, R, vel, angvel)
        rows = [locals_[i].prepare(p_i, v_i, R_i, w_i, t + CLOCK_OFFSET)
                for i, (p_i, R_i, v_i, w_i) in enumerate(ests)]
        Xn = ((np.stack(rows) - om) / os_).astype(np.float32)
        with torch.no_grad():
            lam_pol = base_net(torch.tensor(Xn)).numpy()                    # (N,N): row i = drone i's vector

        # Clean centralized optimum on the TRUE state (the privileged f_central).
        f_central, _, _ = clean.compute_forces(pos, vel, R, angvel, t, bypass_opt=BYPASS_OPT)

        # Phase A: per drone, build obs (30-D) + fixed target + record base-only f_local.
        obs_row = np.zeros((N, 30), dtype=np.float32)
        f_local_all = np.zeros((N, 3))
        dstar_all = np.zeros((N, 3))
        for i in range(N):
            p_i, R_i, v_i, w_i = ests[i]
            f_full_i = locals_[i].finalize(lam_pol[i])                      # rolls history; == f_g+f_lambda
            sl = slice(3 * i, 3 * i + 3)
            f_g_i = (locals_[i]._G_pinv @ locals_[i]._w_d)[sl]              # load-serving part
            f_local_i = f_full_i[sl]
            f_lam_i = f_local_i - f_g_i                                     # nullspace part = (N lambda)_i

            load18 = np.concatenate([p_i, R_i.flatten(order="C"), v_i, w_i])
            own = obs42[np.r_[18 + 3 * i: 18 + 3 * i + 3,
                              18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]]
            obs_row[i] = np.concatenate([load18, own, f_g_i, f_lam_i])

            dstar_i = f_central[sl] - f_local_i                            # privileged residual target
            y_i = _cap(dstar_i, f_local_i)                                 # LABEL clipped to deployable cap
            X_rows.append(obs_row[i]); Y_rows.append(y_i.astype(np.float32))
            dstar[i].append(np.linalg.norm(dstar_i))                       # UNclipped magnitude (true demand)
            f_local_all[i] = f_local_i
            dstar_all[i] = dstar_i                                         # UNclipped for drive (beta=1 -> f_central)
            if np.linalg.norm(dstar_i) > RESIDUAL_CAP * np.linalg.norm(f_local_i):
                cap_hit += 1

        # Phase B: policy residual (capped exactly like the env), then beta-mix the drive.
        df_policy = np.zeros((N, 3))
        if resnet is not None:
            On = ((obs_row - res_om) / res_os).astype(np.float32)
            with torch.no_grad():
                raw = resnet(torch.tensor(On)).numpy() * RESIDUAL_SCALE     # (N,3)
            for i in range(N):
                df_policy[i] = _cap(raw[i], f_local_all[i])

        f_applied = np.zeros(3 * N)
        for i in range(N):
            df = beta * dstar_all[i] + (1.0 - beta) * df_policy[i]
            f_applied[3 * i: 3 * i + 3] = f_local_all[i] + df
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        load_hist.append(pos.copy())
        ref_hist.append(get_reference_trajectory(t)[0].copy())

        ff = LLC_ALPHA * f_applied + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    dstar = [np.array(d) for d in dstar]
    dvel = [np.array(v) for v in dvel]
    load = np.array(load_hist); ref = np.array(ref_hist)
    err = np.linalg.norm(load - ref, axis=1)
    diag = {
        "delays": delays,
        "track_mean": float(err.mean()), "track_max": float(err.max()),
        "dstar_mean": float(np.mean([d.mean() for d in dstar])),
        "dstar_max": float(max(d.max() for d in dstar)),
        "cap_hit_frac": cap_hit / (N * len(load)),
        "load": load, "ref": ref, "dstar": dstar, "dvel": dvel,
        "t": np.arange(len(load)) * DT,
    }
    return (np.asarray(X_rows, dtype=np.float32),
            np.asarray(Y_rows, dtype=np.float32), diag)


def train(X, Y):
    """Regress obs33 -> delta_f* (MSE). Fresh normalization, 80/20 split, best-val ckpt."""
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

    net = Actor(obs_dim=X.shape[1], act_dim=Y.shape[1]).to(DEVICE)
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
    """Per-iteration: required correction magnitude ||delta_f*|| + velocity norms."""
    fig1, ax1 = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(ax1):
        ax.plot(diag["t"], diag["dstar"][i], "c", lw=1.0)
        ax.set_ylabel(f"$\\|\\delta f^*_{i+1}\\|$ (N)"); ax.grid(True)
    ax1[-1].set_xlabel("Time (s)")
    fig1.suptitle(f"Required residual correction |delta_f*| — {label}")

    plt.figure()
    for i in range(N):
        plt.plot(diag["t"], diag["dvel"][i], label=f"Drone {i+1}")
    plt.axhline(EPS, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title(f"Drone velocity norms — {label}"); plt.legend(); plt.grid(True)

    if PAUSE_SEC > 0:
        plt.show(block=False); plt.pause(PAUSE_SEC); plt.close("all")
    else:
        plt.show()


def show_compare(base_diag, res_diag):
    """Load tracking + velocity: base-only vs base+warm-started residual (same desync seq)."""
    t = base_diag["t"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(t, base_diag["ref"][:, k], "k--", lw=2, label="reference")
        ax.plot(t, base_diag["load"][:, k], "r", alpha=0.8, label="base only")
        ax.plot(t, res_diag["load"][:, k], "b", label="base + residual")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True)
        if k == 0:
            ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"Load tracking under desync — warm-start residual\n"
                 f"base {base_diag['track_mean']:.4f} -> residual {res_diag['track_mean']:.4f} (mean err)")

    plt.figure()
    for i in range(N):
        plt.plot(t, res_diag["dvel"][i], label=f"Drone {i+1}")
    plt.axhline(EPS, ls="--", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title("Drone velocity norms — base + residual"); plt.legend(); plt.grid(True)

    if PAUSE_SEC > 0:
        plt.show(block=False); plt.pause(PAUSE_SEC); plt.close("all")
    else:
        plt.show()


def main():
    # Frozen base policy + its normalization.
    ck = torch.load(BASE_POLICY, map_location="cpu", weights_only=False)
    base_net = Actor(obs_dim=ck["obs_mean"].shape[1], act_dim=N)
    base_net.load_state_dict(ck["state_dict"]); base_net.eval()
    om, os_ = ck["obs_mean"].astype(np.float32), ck["obs_std"].astype(np.float32)

    rng = np.random.default_rng(SEED)          # advances across rollouts -> fresh noise + delays each iter
    resnet, res_om, res_os = None, None, None
    D_X, D_Y = None, None

    for k, beta in enumerate(BETAS, 1):
        new_X, new_Y, diag = rollout(base_net, om, os_, resnet, res_om, res_os, beta, rng)
        show_diag(diag, f"iter {k}  beta {beta:.2f}  delays {diag['delays'].tolist()}")

        D_X = new_X if D_X is None else np.concatenate([D_X, new_X], axis=0)
        D_Y = new_Y if D_Y is None else np.concatenate([D_Y, new_Y], axis=0)
        state, res_om, res_os, best_va, var_d = train(D_X, D_Y)
        resnet = Actor(obs_dim=D_X.shape[1], act_dim=3)
        resnet.load_state_dict(state); resnet.eval()

        print(f"iter {k}  beta {beta:.2f}  delays {diag['delays'].tolist()}  |  "
              f"track {diag['track_mean']:.4f}  |delta_f*| mean {diag['dstar_mean']:.3f} "
              f"max {diag['dstar_max']:.3f}  cap-hit {diag['cap_hit_frac']*100:.1f}%  |  "
              f"fit MSE {best_va:.5f} (Var {var_d:.4f})  |  dataset {len(D_X)}")

    torch.save({"state_dict": resnet.state_dict(), "obs_mean": res_om, "obs_std": res_os},
               "residual_warmstart.pt")
    print("\nsaved residual_warmstart.pt")

    # Fair before/after on the SAME desync sequence (reset rng to a fixed eval seed).
    # beta=0 both: base-only = f_local (no correction) vs base + pure residual (deployment).
    eval_seed = 12345
    _, _, base_diag = rollout(base_net, om, os_, None, None, None, 0.0, np.random.default_rng(eval_seed))
    _, _, res_diag = rollout(base_net, om, os_, resnet, res_om, res_os, 0.0, np.random.default_rng(eval_seed))
    print(f"\nEVAL (seed {eval_seed}):  base-only track mean {base_diag['track_mean']:.4f} "
          f"max {base_diag['track_max']:.4f}   ->   base+residual mean {res_diag['track_mean']:.4f} "
          f"max {res_diag['track_max']:.4f}")
    show_compare(base_diag, res_diag)
    print("Next: MAPPO fine-tune (Stage 2) warm-started from residual_warmstart.pt.")


if __name__ == "__main__":
    main()
