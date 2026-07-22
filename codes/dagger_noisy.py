"""
dagger_noisy.py — noisy DAgger for the pR_dot base policy (desync robustness).

The filtered base policy is smooth+correct at ZERO noise, but under desync its lambda
self-feedback loop (lambda_{t-1} -> pR_dot -> lambda_t) gets re-excited and jitters
(jitter_diag: pR_dot 34x, velocity 21x). The filter fixed the *reconstruction's* noise
amplification; it can't fix *policy* sensitivity, because the policy was only ever
trained at zero noise.

Fix = train the base WITH the desync on, so it learns to DENOISE: map a noisy/delayed
reconstructed pR_dot to the CLEAN-optimal lambda. Per step, in the DECENTRALIZED
structure (each drone on its OWN noisy/delayed load view, exactly like deployment):

    input  (per drone) : reconstruct_lp(noisy view) -> [clock, pR_dot, lambda_{t-1}]
    label              : the CLEAN optimizer's lambda on the TRUE state (one vector)
    drive (beta-mix)   : lambda_applied_i = beta*lambda_clean + (1-beta)*lambda_policy_i

Every drone (different noisy view) is trained toward the SAME clean lambda -> it learns
to average out the noise, which also gives implicit coordination. beta 0.9->0 slides the
state distribution from the clean trajectory to the deployment (self-fed policy) one.

Spec (bounded / realistic; escalate only if it holds):
  - pos+vel sensing noise (fixed std, AR(1)); FRESH rng each rollout (no memorized seq).
  - per-drone control delay sampled from {1,2} steps each rollout; NO clock offset.
The honest limit (a persistently-delayed/biased view can't be fully recovered) is left
for the RL residual to mop up. Saves il_actor_prdot_noisy.pt.
"""

import copy
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from controller import error_calculation, get_reference_trajectory
from networks import Actor
from collect_il_data import read_params, N, DT, T_END, EPS, PHASES, LLC_ALPHA, FZ
from collect_prdot_data import LAM0, RECON_ALPHA
from residual_marl_env import LocalModelAgent

BYPASS_OPT = False   # clean adaptive optimizer as the label source

# --- desync spec (bounded, realistic; start mild) ---
POS_NOISE = 0.03
VEL_NOISE = 0.10
ANGVEL_NOISE = 0.0
ROT_NOISE = 0.0
NOISE_CORR = 0.995
DELAY_CHOICES = (1, 2)       # per-drone control delay (steps), sampled per rollout
CLOCK_OFFSET = 0.0           # start with none

# beta schedule (drive: beta*clean + (1-beta)*policy). 1 = clean trajectory, 0 = deployment.
BETAS = [0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]

EPOCHS = 250
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.2
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAUSE_SEC = 20.0


def project_SO3(M):
    U, _, Vt = np.linalg.svd(M)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


class DesyncSensor:
    """Per-drone noisy/delayed load views — same model as ResidualMARLEnv._update_estimates."""

    def __init__(self, n, delays, rng):
        self.n = n
        self.rng = rng
        self.delays = np.asarray(delays, dtype=int)
        self.buffer = deque(maxlen=int(self.delays.max()) + 1)
        self.noise_pos = np.zeros((n, 3))
        self.noise_vel = np.zeros((n, 3))
        self.noise_angvel = np.zeros((n, 3))
        self.noise_rot = np.zeros((n, 3, 3))

    def _ar1(self, prev, sigma):
        if sigma <= 0:
            return prev
        eps = self.rng.standard_normal(prev.shape)
        return NOISE_CORR * prev + np.sqrt(1.0 - NOISE_CORR ** 2) * sigma * eps

    def update(self, pos, R, vel, angvel):
        """Push the true state, advance noise, return per-drone (p, R, v, w) estimates."""
        self.buffer.append((pos, R, vel, angvel))
        self.noise_pos = self._ar1(self.noise_pos, POS_NOISE)
        self.noise_vel = self._ar1(self.noise_vel, VEL_NOISE)
        self.noise_angvel = self._ar1(self.noise_angvel, ANGVEL_NOISE)
        self.noise_rot = self._ar1(self.noise_rot, ROT_NOISE)
        ests = []
        for i in range(self.n):
            d = min(int(self.delays[i]), len(self.buffer) - 1)
            p0, R0, v0, w0 = self.buffer[-1 - d]
            p = p0 + self.noise_pos[i]
            v = v0 + self.noise_vel[i]
            w = w0 + self.noise_angvel[i]
            Rn = project_SO3(R0 + self.noise_rot[i]) if ROT_NOISE > 0 else R0
            ests.append((p, Rn, v, w))
        return ests


def rollout(net, om, os_, beta, rng):
    """One decentralized noisy episode. Policy on per-drone noisy views, labels from the
    clean optimizer (true state), beta-mixed driving. Returns (M,30) inputs, (M,N) labels."""
    delays = rng.integers(DELAY_CHOICES[0], DELAY_CHOICES[1] + 1, size=N)
    env = FMUPlantEnv(n_carriers=N, step_size=DT, end_time=T_END)
    obs42, _ = env.reset()
    J, Bb, m, L0 = read_params(env)
    clean = ClassicalAgent(N, DT, PHASES, EPS, L0, m, J, Bb)   # clean expert on the TRUE state
    locals_ = [LocalModelAgent(N, DT, PHASES, EPS, L0, m, J, Bb, RECON_ALPHA) for _ in range(N)]
    sensor = DesyncSensor(N, delays, rng)

    prev_f = np.array([0.0, 0.0, FZ] * N)
    X_rows, Y_rows = [], []
    lam_pol_hist = [[] for _ in range(N)]
    lam_exp_hist = [[] for _ in range(N)]
    dvel = [[] for _ in range(N)]
    load_hist, ref_hist = [], []

    t = 0.0
    while t < T_END - 1e-9:
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        vel, angvel = obs42[12:15], obs42[15:18]

        # Per-drone noisy/delayed views -> per-drone policy input rows.
        ests = sensor.update(pos, R, vel, angvel)
        rows = [locals_[i].prepare(p_i, v_i, R_i, w_i, t + CLOCK_OFFSET)
                for i, (p_i, R_i, v_i, w_i) in enumerate(ests)]
        Xn = ((np.stack(rows) - om) / os_).astype(np.float32)
        with torch.no_grad():
            lam_pol = net(torch.tensor(Xn)).numpy()            # (N, N): row i = drone i's whole vector

        # Clean expert label on the TRUE state (one lambda vector for all drones).
        _, _, lam_clean = clean.compute_forces(pos, vel, R, angvel, t, bypass_opt=BYPASS_OPT)

        # Record (noisy input -> clean label), beta-mix per drone, drive.
        f_applied = np.zeros(3 * N)
        for i in range(N):
            X_rows.append(rows[i])
            Y_rows.append(lam_clean)
            lam_app = beta * lam_clean + (1.0 - beta) * lam_pol[i]
            f_full = locals_[i].finalize(lam_app)              # own w_d/G+/N; rolls history
            f_applied[3 * i: 3 * i + 3] = f_full[3 * i: 3 * i + 3]
            lam_pol_hist[i].append(lam_pol[i][i])
            lam_exp_hist[i].append(lam_clean[i])
            dvel[i].append(np.linalg.norm(obs42[18 + 3 * N + 3 * i: 18 + 3 * N + 3 * i + 3]))
        load_hist.append(pos.copy())
        ref_hist.append(get_reference_trajectory(t)[0].copy())

        ff = LLC_ALPHA * f_applied + (1 - LLC_ALPHA) * prev_f
        deriv = (ff - prev_f) / DT
        prev_f = ff.copy()
        obs42, *_ = env.step(np.concatenate([ff, deriv]))
        t += DT
    env.close()

    lam_pol_hist = [np.array(l) for l in lam_pol_hist]
    lam_exp_hist = [np.array(l) for l in lam_exp_hist]
    dvel = [np.array(v) for v in dvel]
    load = np.array(load_hist); ref = np.array(ref_hist)
    buzz = float(np.mean([np.mean(np.abs(np.diff(l))) for l in lam_pol_hist]))
    diag = {
        "buzz": buzz, "delays": delays,
        "vmin": float(min(v.min() for v in dvel)),
        "vmean": float(np.mean([v.mean() for v in dvel])),
        "track_mean": float(np.linalg.norm(load - ref, axis=1).mean()),
        "track_max": float(np.linalg.norm(load - ref, axis=1).max()),
        "lam_pol": lam_pol_hist, "lam_exp": lam_exp_hist, "dvel": dvel,
        "t": np.arange(len(load)) * DT,
    }
    return (np.asarray(X_rows, dtype=np.float32),
            np.asarray(Y_rows, dtype=np.float32), diag)


def train(X, Y):
    """Retrain from scratch on the aggregated set. Fresh normalization, best-val checkpoint."""
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
    fig1, ax1 = plt.subplots(N, 1, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(ax1):
        ax.plot(diag["t"], diag["lam_exp"][i], "k", lw=1.0, label="clean expert")
        ax.plot(diag["t"], diag["lam_pol"][i], "m", lw=1.0, alpha=0.8, label="policy (noisy view)")
        ax.set_ylabel(f"$\\lambda_{i+1}$"); ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    ax1[-1].set_xlabel("Time (s)"); fig1.suptitle(f"Clean vs policy lambda — {label}")

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


def main():
    # Warm-start from the clean DAgger'd policy + its (clean) dataset.
    ckpt = torch.load("il_actor_prdot_dagger.pt", map_location="cpu", weights_only=False)
    net = Actor(obs_dim=ckpt["obs_mean"].shape[1], act_dim=N)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    om, os_ = ckpt["obs_mean"].astype(np.float32), ckpt["obs_std"].astype(np.float32)

    data = np.load("prdot_dataset.npz")
    D_X = data["X"].astype(np.float32)
    D_Y = data["Y"].astype(np.float32)

    rng = np.random.default_rng(SEED)   # advances across rollouts -> fresh noise + delays each iter
    buzz_curve = []
    for k, beta in enumerate(BETAS, 1):
        new_X, new_Y, diag = rollout(net, om, os_, beta, rng)
        show_diag(diag, f"iter {k}  beta {beta:.2f}  delays {diag['delays'].tolist()}")

        D_X = np.concatenate([D_X, new_X], axis=0)
        D_Y = np.concatenate([D_Y, new_Y], axis=0)
        state, om, os_, best_va, var_lam = train(D_X, D_Y)
        net = Actor(obs_dim=D_X.shape[1], act_dim=N)
        net.load_state_dict(state); net.eval()

        buzz_curve.append(diag["buzz"])
        print(f"iter {k}  beta {beta:.2f}  delays {diag['delays'].tolist()}  |  "
              f"rollout buzz {diag['buzz']:.4f}  vmin {diag['vmin']:.3f}  "
              f"track {diag['track_mean']:.4f}  |  retrain MSE {best_va:.4f}  |  dataset {len(D_X)}")

    torch.save({"state_dict": net.state_dict(), "obs_mean": om, "obs_std": os_},
               "il_actor_prdot_noisy.pt")
    print("\nsaved il_actor_prdot_noisy.pt")

    plt.figure()
    plt.plot(range(1, len(buzz_curve) + 1), buzz_curve, "o-")
    plt.xlabel("noisy-DAgger iteration"); plt.ylabel("policy action buzz  (mean |d lambda/step|)")
    plt.title("Closed-loop lambda buzz across noisy DAgger"); plt.grid(True)
    plt.show()
    print("Next: point ResidualMARLEnv policy_path at il_actor_prdot_noisy.pt and run demo_desync.")


if __name__ == "__main__":
    main()
