"""
collect_prdot_data.py — data for the pR_dot policy (the optimizer's true I/O, distilled).

The optimizer's whole computation exists to produce ONE thing per drone: the carrier
velocity v_Ri (Eq. 22), which it regulates against the epsilon floor. So instead of
feeding the net the raw ingredients (G, N, w_d, ...) we feed that distilled quantity
directly, plus the clock for phase and the previous lambda for the amplitude anchor:

    input :  [ clock (14) ,  pR_dot (n*3) ,  lambda_{t-1} (n) ]
    output:  lambda[n]                              # the optimizer's full vector

Roles: clock -> oscillation phase; pR_dot -> amplitude feedback (the constraint
variable); lambda_{t-1} -> the smoothness/anchor state (pR_dot is lossy about it, so
the net cannot recover it otherwise).

pR_dot is reconstructed EXACTLY as the optimizer computes v_Ri (analytic, unfiltered
force and analytic derivative, Eq. 19+22, tension floor) — NOT from the LLC-filtered
plant force — so the net learns on the same quantity the optimizer keyed its lambda on.
It is one-step lagged: built from lambda_{t-1} (and lambda_dot_{t-1} via finite
difference of the lambda history, so collection and deploy compute it identically).

reconstruct/build_input live here so deploy_prdot.py imports the identical helpers —
collection and deployment cannot diverge.
"""

import numpy as np
import matplotlib.pyplot as plt
from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from optimizer import calculate_grasp_and_nullspace, cable_force_calculation
from controller import error_calculation, get_reference_trajectory
from collect_il_data import clock_features, read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from trajectories import train_set, custom_set

BYPASS_OPT = False   # adaptive optimizer (the real sweeping target)
N_TRAJ = 50          # GENERALIZATION: number of sampled quintic-pose trajectories to collect over
                     #   (train_set, fixed seed). Was effectively 1 (single default trajectory).

# Reconstruction mode toggle: False = low-pass (reconstruct_lp, the deployed base);
# True = the original finite-difference `reconstruct`. Suffixes all outputs with
# _analytic so both variants coexist. Drives the Reconstructor below, so the
# collect/train/deploy/dagger loops stay mode-agnostic.
ANALYTIC = True
SUFFIX = "_analytic" if ANALYTIC else ""
OUT = f"prdot_dataset{SUFFIX}.npz"

# Optimizer's initial lambda (matches ClassicalAgent.reset's seed): A0*cos(xi0*0 + phases).
LAM0 = 1.2 * np.cos(np.asarray(PHASES))

# ---- lambda-HISTORY input (F1 phase-inertia / hidden-state memory) --------------------
# The net sees its last LAM_HIST APPLIED lambdas (newest-first), not just lambda_{t-1}. Why:
# a continuous recent trajectory can't jump to anti-phase without a discontinuity the net never
# saw in training -> phase INERTIA that resists the closed-loop phase slip; and the window makes
# the optimizer's hidden A/xi state observable. Cold start = analytic continuation of the init
# sinusoid to NEGATIVE time: during the hold the real lambda IS A0*cos(XI0*t + PHASES), so the
# fake past is a SEAMLESS extension of it (not made-up data). Shared by collect/dagger/deploy so
# the buffer is built identically -> no train/deploy mismatch.
XI0, A0 = 2.0, 1.2            # optimizer's initial (frequency, amplitude); LAM0 = A0*cos(XI0*0+PHASES)
LAM_HIST = 10                 # number of past applied-lambda rows fed to the net


def init_lam_history(n_hist=LAM_HIST):
    """Newest-first cold-start buffer (n_hist, N): lambda_{-1..-n_hist} on the init sinusoid,
    lambda_{-k} = A0*cos(XI0*(-k*DT) + PHASES). Seamless continuation of the real early lambda."""
    ph = np.asarray(PHASES)
    return np.stack([A0 * np.cos(XI0 * (-(k + 1) * DT) + ph) for k in range(n_hist)])


def push_lam(buf, lam):
    """Push applied `lam` (N,) onto the newest-first buffer, drop the oldest -> new (n_hist, N)."""
    return np.vstack([np.asarray(lam)[None, :], buf[:-1]])

# ---- F1 policy capacity (shared by train_prdot / dagger_prdot; saved in ckpts) --------
PRDOT_HIDDEN = (256, 256)     # was (128,128); loaders read "hidden" from the ckpt

# ---- IN-LOOP output-lambda EMA (shared by dagger_prdot rollout AND deploy_prdot) -------
# Soft warm-start CONTINUITY: applied_lambda = a*raw + (1-a)*applied_prev, a = DT/(tau+DT).
# The optimizer stays stable in the same reconstructed-vR loop because its warm-started solve
# forbids lambda from jumping; the memoryless net lost that, so we re-impose it here. Trained
# IN the DAgger rollout (not bolted on) so there's no train/deploy mismatch. Light (tau=0.01 ->
# a=0.5) = minimal lag, just damps step-to-step jitter (the div-by-dt amplifier's worst band).
# None = off; raise tau only if it doesn't stabilize.
LAM_LP_TAU = 0.01

# ---- ACTIVITY bins + two-level HARDNESS curation (shared by collect / train / dagger) --
# Semantic bins from the REFERENCE kinematics (what the load is doing), so curation can keep
# a diverse mix and weight harder regimes more. Bins: 0 loiter (parked), 1 cruise (steady
# move), 2 transition (accel/decel ramp).
V_LO = 0.05          # ref speed below this -> loiter (m/s)
A_HI = 0.02          # ref |accel| above this -> transition (m/s^2)
BIN_NAMES = {0: "loiter", 1: "cruise", 2: "transition"}
P_MIN = 0.15         # TRUE overall keep floor per bin: no activity regime, however easy, drops
                     #   below this fraction (diversity guarantee). p_bin IS the bin's kept share.
RECENCY_GAMMA = 0.85 # DAgger recency: a sample from g iters ago is weighted gamma**g -> recent
                     #   (more on-policy, lower-beta) corrections outweigh stale ones (1.0 = off).

# Decile weights normalized to MEAN 1, so multiplying by p_bin makes p_bin the bin's overall kept
# fraction (not p_bin*0.55). Shape kept: hardest tenth ~1.82x the mean, easiest ~0.18x.
_DECILE_W = (np.arange(10) + 1) / 10.0
_DECILE_NORM = _DECILE_W / _DECILE_W.mean()          # mean 1.0, range [0.18, 1.82]


def activity_bin(speed, accel):
    """Classify one step by reference kinematics -> bin id (0 loiter / 1 cruise / 2 transition)."""
    if speed < V_LO:
        return 0
    if accel > A_HI:
        return 2
    return 1


def recency_weights(ages, gamma=RECENCY_GAMMA):
    """gamma**(newest_age - age) per sample: recent DAgger iters ~1, old ones decay toward 0."""
    ages = np.asarray(ages, dtype=float)
    if len(ages) == 0:
        return ages
    return gamma ** (ages.max() - ages)


def keep_probs(H, bins, p_min=P_MIN):
    """Two-level keep-probability per sample (the graduated-decile scheme).

    Level 1 (bin): p_bin = max(p_min, mean_hardness(bin) / max_bin_mean_hardness) -> the bin's
      OVERALL kept fraction; harder activity regimes keep more, none below p_min.
    Level 2 (within bin): rank by hardness into deciles; decile d keeps p_bin * _DECILE_NORM[d]
      (mean-1 normalized), so the bin's mean keep == p_bin while the hardest tenth is favored.
    Returns float array in [0,1].
    """
    H = np.asarray(H, dtype=float)
    bins = np.asarray(bins)
    probs = np.zeros(len(H))
    ids = np.unique(bins)
    bin_score = {b: H[bins == b].mean() if np.any(bins == b) else 0.0 for b in ids}
    smax = max(bin_score.values()) + 1e-12
    for b in ids:
        idx = np.where(bins == b)[0]
        n = len(idx)
        p_bin = max(p_min, bin_score[b] / smax)
        order = np.argsort(H[idx])                    # ascending: easy -> hard
        dec = np.empty(n, dtype=int)
        dec[order] = (np.arange(n) * 10) // max(n, 1)  # 0..9
        probs[idx] = np.clip(p_bin * _DECILE_NORM[dec], 0.0, 1.0)
    return probs


def curate(X, Y, H, bins, rng, p_min=P_MIN, weights=None):
    """Stochastic keep-mask from keep_probs (optionally x recency weights) -> a hardness-weighted,
    bin-diverse, recency-biased subset."""
    p = keep_probs(H, bins, p_min)
    if weights is not None:
        p = np.clip(p * weights, 0.0, 1.0)
    return rng.random(len(p)) < p


def cap_indices(H, bins, max_n, rng, p_min=P_MIN, weights=None):
    """Indices to RETAIN so the aggregate stays <= max_n: highest keep-prob (x recency) wins, ties
    broken randomly (within-decile diversity). Returns all indices if already under cap."""
    n = len(H)
    if n <= max_n:
        return np.arange(n)
    score = keep_probs(H, bins, p_min)
    if weights is not None:
        score = score * weights
    score = score + 1e-6 * rng.random(n)             # noise = random tie-break
    return np.argsort(score)[::-1][:max_n]

# Reconstruction low-pass: OUR estimate of the drone LLC time const (MUST match the
# env's recon_tau). Kills the finite-difference noise amplification; tau < plant's 0.2
# trades a little jitter for less lag.
RECON_TAU = 0.1
RECON_ALPHA = DT / (RECON_TAU + DT)


def reconstruct(R, vL, omega, w_d, lam_prev, lamdot_prev,
                prev_G_pinv, prev_Nmat, prev_w_d, Bb, L0, dt):
    """Carrier velocities v_R (n,3), computed EXACTLY as the optimizer's v_Ri (Eq. 22).

    Analytic, filter-independent: f = G+ w_d + N lambda_{t-1}, and its analytic
    derivative f_dot = e_total + N_dot lambda_{t-1} + N lambda_dot_{t-1} (Eq. 19), with
    the sqrt(||f||^2 + 1e-6) tension floor. One-step lagged via lambda_{t-1}.

    Also returns the current (G_pinv, N) so the caller can roll the derivative history.
    First step: pass prev_*=None -> zero finite-difference derivatives (matches optimizer).
    """
    _, G_pinv, Nmat = calculate_grasp_and_nullspace(R, Bb, N)

    if prev_G_pinv is None:
        G_pinv_dot = np.zeros_like(G_pinv)
        N_dot = np.zeros_like(Nmat)
        w_d_dot = np.zeros_like(w_d)
    else:
        G_pinv_dot = (G_pinv - prev_G_pinv) / dt
        N_dot = (Nmat - prev_Nmat) / dt
        w_d_dot = (w_d - prev_w_d) / dt

    e_total = G_pinv_dot @ w_d + G_pinv @ w_d_dot            # Eq. 20
    f = G_pinv @ w_d + Nmat @ lam_prev                       # analytic cable force
    f_dot = e_total + N_dot @ lam_prev + Nmat @ lamdot_prev  # Eq. 19

    vR = np.zeros((N, 3))
    for i in range(N):
        f_i = f[3 * i: 3 * i + 3]
        fd_i = f_dot[3 * i: 3 * i + 3]
        T = np.sqrt(f_i @ f_i + 1e-6)                        # tension floor, as optimizer
        q = f_i / T
        vLi = vL + R @ np.cross(omega, Bb[i])
        Pi = np.eye(3) - np.outer(q, q)
        vR[i] = vLi + (L0 / T) * Pi @ fd_i
    return vR, G_pinv, Nmat


def reconstruct_lp(R, vL, omega, w_d, lam_prev, prev_f_lp, Bb, L0, dt, alpha):
    """Noise-robust reconstruct: low-pass the LOCAL base force, then differentiate.

    Kills the finite-difference noise amplification of `reconstruct` (which divides
    noisy w_d / lambda differences by dt -> x1/dt gain). Here the base force
    f = G+ w_d + N*lambda_{t-1} is low-passed with the same LLC filter the plant uses,
    and its SMOOTH derivative drives v_Ri. Fully local -- the base force needs only the
    drone's own load view + own lambda, NO neighbour delta_f -- so it's decentralization-
    legal. Returns v_R (n,3), the updated filtered force f_lp, and G+/N (for the caller's
    force calc, so it isn't recomputed).
    """
    _, G_pinv, Nmat = calculate_grasp_and_nullspace(R, Bb, N)
    f = G_pinv @ w_d + Nmat @ lam_prev                      # base force (all N)
    if prev_f_lp is None:
        f_lp = f.copy()
        f_dot = np.zeros_like(f)
    else:
        f_lp = alpha * f + (1.0 - alpha) * prev_f_lp        # low-pass -> kills the noise
        f_dot = (f_lp - prev_f_lp) / dt                     # derivative of a SMOOTH signal

    vR = np.zeros((N, 3))
    for i in range(N):
        f_i = f_lp[3 * i: 3 * i + 3]
        fd_i = f_dot[3 * i: 3 * i + 3]
        T = np.sqrt(f_i @ f_i + 1e-6)
        q = f_i / T
        vLi = vL + R @ np.cross(omega, Bb[i])
        Pi = np.eye(3) - np.outer(q, q)
        vR[i] = vLi + (L0 / T) * Pi @ fd_i
    return vR, f_lp, G_pinv, Nmat


def build_input(t, vR, lam):
    """The net's input row: [clock(14), pR_dot(n*3), lambda-block]. `lam` is EITHER lambda_{t-1}
    (N,) [legacy single-step; still used by F2's residual env] OR a newest-first history buffer
    (LAM_HIST, N) [F1 memory] — flattened either way, so the input dim follows whichever is passed."""
    return np.concatenate([clock_features(t), vR.flatten(), np.asarray(lam).flatten()])


class Reconstructor:
    """Per-step pR_dot with internal state, so collect/deploy/dagger loops are identical
    regardless of mode. ANALYTIC=True -> finite-difference `reconstruct` (needs prev G+/N/
    w_d + lambda_dot); ANALYTIC=False -> `reconstruct_lp` (needs prev filtered force).

    Usage:  vR = recon(R, vL, omega, w_d, lam_prev);  ... ;  recon.roll(lam_applied)
    """

    def __init__(self, Bb, L0, dt):
        self.Bb, self.L0, self.dt = Bb, L0, dt
        self.prev_G = self.prev_N = self.prev_wd = None      # analytic history
        self.prev_prev_lam = LAM0.copy()                     # analytic lambda_{t-2}
        self.prev_f_lp = None                                # lp history

    def __call__(self, R, vL, omega, w_d, lam_prev):
        if ANALYTIC:
            lamdot = (lam_prev - self.prev_prev_lam) / self.dt
            vR, G, Nm = reconstruct(R, vL, omega, w_d, lam_prev, lamdot,
                                    self.prev_G, self.prev_N, self.prev_wd, self.Bb, self.L0, self.dt)
            self._G, self._N, self._wd, self._lam_prev = G, Nm, w_d, lam_prev
        else:
            vR, f_lp, _, _ = reconstruct_lp(R, vL, omega, w_d, lam_prev, self.prev_f_lp,
                                            self.Bb, self.L0, self.dt, RECON_ALPHA)
            self._f_lp = f_lp
        return vR

    def roll(self, lam_applied):
        """Advance the internal history with the lambda that was actually applied."""
        if ANALYTIC:
            self.prev_G, self.prev_N, self.prev_wd = self._G, self._N, self._wd
            self.prev_prev_lam = self._lam_prev
        else:
            self.prev_f_lp = self._f_lp


def rollout(env, agent, Bb, L0, traj):
    """One full-episode expert rollout on reference `traj` (None -> default trajectory).
    Fresh per-episode state (env + agent are reused across trajectories, reset here).
    Returns (X_rows, Y_rows, hist)."""
    obs42, _ = env.reset()
    agent.reset()

    prev_f = np.array([0.0, 0.0, FZ] * N)           # applied force (for plant filtering only)
    prev_lam = LAM0.copy()                          # lambda_{t-1} (for the pR_dot reconstruction)
    lam_buf = init_lam_history()                    # newest-first last-LAM_HIST applied lambda (input)
    prev_vLd = None                                 # ref velocity_{t-1} (for accel -> activity bin)
    recon = Reconstructor(Bb, L0, DT)
    X_rows, Y_rows, H_rows, bin_rows = [], [], [], []

    # Histories for the diagnostic plots.
    t_hist, load_hist, ref_hist = [], [], []
    dpos = [[] for _ in range(N)]
    dvel_true = [[] for _ in range(N)]              # plant drone velocity norm
    prnorm = [[] for _ in range(N)]                 # reconstructed pR_dot norm (the feature)
    lam_hist = [[] for _ in range(N)]               # optimizer target lambda

    t = 0.0
    while t < T_END - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        # w_d ONCE (advances integrators once); reused for both input and target.
        ep, eR, ev, ew = error_calculation(pos, vel, R, angvel, t, traj)
        w_d = agent.wrench_control(ep, eR, ev, ew, angvel)

        # Input: pR_dot (mode set by ANALYTIC) from lambda_{t-1} + last-LAM_HIST applied lambda.
        vR = recon(R, vel, angvel, w_d, prev_lam)
        X_rows.append(build_input(t, vR, lam_buf))

        # Target: the optimizer's full lambda vector (same w_d, so no double integration).
        lam, _ = agent.optimize(t, R, vel, angvel, w_d, bypass=BYPASS_OPT)
        Y_rows.append(lam)
        f_full, _ = cable_force_calculation(R, Bb, w_d, lam, N)

        # Activity bin (ref kinematics) + BC-hardness proxy = target step change ||lam - lam_{t-1}||
        # (large where the optimizer lambda is NOT persistence -> the informative samples).
        ref = get_reference_trajectory(t, traj)
        v_Ld = ref[1]
        accel = 0.0 if prev_vLd is None else float(np.linalg.norm(v_Ld - prev_vLd) / DT)
        bin_rows.append(activity_bin(float(np.linalg.norm(v_Ld)), accel))
        H_rows.append(float(np.linalg.norm(lam - prev_lam)))
        prev_vLd = v_Ld.copy()

        # Record.
        t_hist.append(t)
        load_hist.append(pos.copy())
        ref_hist.append(ref[0].copy())
        for i in range(N):
            dpos[i].append(obs42[18 + 3 * i: 18 + 3 * i + 3].copy())
            dvel_true[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
            prnorm[i].append(np.linalg.norm(vR[i]))
            lam_hist[i].append(lam[i])

        # Step plant (filtered force).
        ff = LLC_ALPHA * f_full + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))

        # Roll histories with the APPLIED lambda (here = the optimizer's lambda).
        recon.roll(lam)
        prev_lam = lam.copy()
        lam_buf = push_lam(lam_buf, lam)
        t += DT

    hist = dict(t=np.array(t_hist), load=np.array(load_hist), ref=np.array(ref_hist),
                dpos=[np.array(p) for p in dpos],
                dvel_true=[np.array(v) for v in dvel_true],
                prnorm=[np.array(v) for v in prnorm],
                lam=[np.array(l) for l in lam_hist])
    return (np.asarray(X_rows, dtype=np.float32),
            np.asarray(Y_rows, dtype=np.float32),
            np.asarray(H_rows, dtype=np.float32),
            np.asarray(bin_rows, dtype=np.int64), hist)


def collect(n_traj=N_TRAJ, include_default=True):
    """Collect the BC dataset over n_traj GENERALIZED trajectories (from trajectories.train_set),
    plus (include_default) the original DEFAULT straight-line trajectory as an anchor. Reuses ONE
    env + ONE agent (expensive CasADi solver) across all episodes."""
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    env.reset()
    J, Bb, m, L0 = read_params(env)
    agent = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)

    trajs = custom_set() + list(train_set(n_traj))   # 5 solver-engaging customs + n_traj quintics
    if include_default:
        trajs.insert(0, (None, None))          # traj=None -> original straight-line trajectory
    X_all, Y_all, H_all, bin_all, last_hist = [], [], [], [], None
    for k, (traj, p) in enumerate(trajs):
        Xk, Yk, Hk, bk, last_hist = rollout(env, agent, Bb, L0, traj)
        X_all.append(Xk); Y_all.append(Yk); H_all.append(Hk); bin_all.append(bk)
        if traj is None:
            print(f"  traj {k + 1:>3}/{len(trajs)}  DEFAULT straight-line          steps={len(Xk)}")
        elif p.get("kind") == "custom":
            print(f"  traj {k + 1:>3}/{len(trajs)}  CUSTOM {p['name']:<7} solver-engaging  steps={len(Xk)}")
        else:
            print(f"  traj {k + 1:>3}/{len(trajs)}  dpos={np.round(p['pos_delta'], 2)} "
                  f"drot(deg)={np.round(np.rad2deg(p['rot_delta']), 1)}  steps={len(Xk)}")
    env.close()
    return (np.concatenate(X_all), np.concatenate(Y_all),
            np.concatenate(H_all), np.concatenate(bin_all), last_hist)


def plot(hist):
    t = hist["t"]; load = hist["load"]; ref = hist["ref"]

    # 1. Load tracking.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
        ax.plot(t, ref[:, k], "k--", lw=2, label="reference")
        ax.plot(t, load[:, k], "b", label="optimizer")
        ax.set_ylabel(f"{lbl} (m)"); ax.grid(True); ax.legend(loc="upper right")
    axes[2].set_xlabel("Time (s)"); fig.suptitle("Load tracking — optimizer (collection)")

    # 2. Drone velocity norms: reconstructed pR_dot (feature) vs true plant velocity.
    plt.figure()
    for i in range(N):
        c = f"C{i}"
        plt.plot(t, hist["prnorm"][i], c, label=f"Drone {i+1} recon pR_dot")
        plt.plot(t, hist["dvel_true"][i], c, ls="--", alpha=0.5)
    plt.axhline(EPS, ls=":", c="gray", label="epsilon")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
    plt.title("Reconstructed pR_dot (solid) vs true drone vel (dashed)"); plt.legend(); plt.grid(True)

    # 3. Optimizer lambda (the target) per drone.
    fig3, ax3 = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(ax3):
        ax.plot(t, hist["lam"][i], "m")
        ax.set_ylabel(f"$\\lambda_{i+1}$ target"); ax.grid(True)
    ax3[-1].set_xlabel("Time (s)"); fig3.suptitle("Optimizer lambda (regression target)")

    # 4. Drone XY trajectories.
    plt.figure(figsize=(8, 6))
    for i in range(N):
        plt.plot(hist["dpos"][i][:, 0], hist["dpos"][i][:, 1], label=f"Drone {i+1}")
    plt.plot(load[:, 0], load[:, 1], "k--", lw=2, label="Load")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("Drone XY trajectories — optimizer"); plt.legend(); plt.grid(True); plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    X, Y, H, bins, hist = collect()
    np.savez(OUT, X=X, Y=Y, H=H, bins=bins)
    print(f"saved {OUT}   steps={len(X)}   X {X.shape}  Y {Y.shape}")
    for b, name in BIN_NAMES.items():
        m = bins == b
        print(f"  bin {b} {name:>10}: {m.sum():>7} steps  mean hardness {H[m].mean() if m.any() else 0:.4f}")
    print(f"  input = clock(14) + pR_dot({3*N}) + lambda_prev({N})  ->  lambda[{N}]")
    print(f"  lambda range [{Y.min():.3f}, {Y.max():.3f}]  std {Y.std():.3f}")
    for i in range(N):
        pr = hist["prnorm"][i]; tv = hist["dvel_true"][i]
        print(f"  drone {i+1}: recon pR_dot norm mean {pr.mean():.3f}  "
              f"| recon-vs-true RMSE {np.sqrt(np.mean((pr - tv) ** 2)):.4f}")
    plot(hist)
