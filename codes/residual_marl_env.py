"""
residual_marl_env.py
====================
Formulation-2 (residual RL) multi-agent environment — NETWORK-based.

Each drone no longer runs a CasADi optimizer. Instead it runs a LocalModelAgent:
a fully-local replica of the whole system built from the frozen pR_dot policy plus a
kinematic reconstruction of the carriers. Per step, for each drone i (on its OWN,
possibly desynced, load view and offset clock):

    f_base_i  = slice_i( G+ w_d + N lambda )         # LocalModelAgent_i, its own load view
    action_i  = [ delta_lambda(n) , delta_wrench(6) ]                 # TWO-HEAD residual RL
    delta_f_i = slice_i( G+ @ delta_wrench + N @ delta_lambda )       # range fixes load, null fixes traj
    f_cmd_i   = f_base_i + delta_f_i
    -> LLC filter -> plant.step()

The pR_dot net needs every carrier's velocity to run, but the drone observes none of
them: it RECONSTRUCTS all N of them analytically (optimizer-exact v_Ri, Eq. 22) from
its OWN load estimate + geometry + its OWN lambda history — the identical
reconstruct() used to train the policy. So there is ZERO inter-drone communication:
each drone rebuilds the whole system from local sensing alone, runs the net to get the
whole lambda vector, and keeps its own slice. Under desync the per-drone views diverge
-> the replicas disagree -> forces stop cancelling -> the load is disturbed. That is
the F2 phenomenon; the residual delta_f learns to fix it.

All noise/desync defaults to 0. At zero noise + zero residual the four replicas are
identical and this reproduces deploy_prdot (the pR_dot policy) driving the plant.
"""

import functools
from collections import deque
import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

from fmu_plant_env import FMUPlantEnv
from controller import error_calculation
from networks import Actor
from collect_prdot_data import reconstruct_lp, build_input, LAM0, init_lam_history, push_lam, LAM_LP_TAU
from collect_il_data import clock_features


class LocalModelAgent:
    """One drone's local replica of the pR_dot WHOLE-VECTOR policy. Fully local, zero
    communication. Split into two phases so the env can batch the N net forwards:
      prepare()  -> wrench + reconstruct(all N pR_dot from its own load estimate + its own
                    lambda history) + build the net input row (env runs the net).
      finalize() -> distribute forces from its whole lambda vector (reusing prepare's G+/N),
                    keep its own slice, roll history.
    STATEFUL: carries its own lambda / G+ / N / w_d history for the analytic reconstruction
    — exactly deploy_prdot's single loop, one copy per drone."""

    def __init__(self, n, dt, phases, epsilon, L0, mass, J, Bb, recon_alpha):
        self.n = n
        self.dt = dt
        self.phases = np.asarray(phases, dtype=float)
        self.epsilon = epsilon
        self.L0 = L0
        self.mass = mass
        self.inertia = np.asarray(J, dtype=float)
        self.Bb = np.asarray(Bb, dtype=float)

        # Wrench-controller PID gains (identical to the classical controller).
        self.Kp = 5.0 * np.eye(3)
        self.Kv = 2.0 * np.eye(3)
        self.Ki = 0.9 * np.eye(3)
        self.KR = 0.5 * np.eye(3)
        self.Kw = 0.06 * np.eye(3)
        self.KiR = 0.1 * np.eye(3)
        self.g = 9.81
        self.e3 = np.array([0.0, 0.0, 1.0])
        self.recon_alpha = recon_alpha               # reconstruction low-pass (tunable estimate)
        self.reset()

    def reset(self):
        self.intg_ep = np.zeros(3)
        self.intg_eR = np.zeros(3)
        # Local history for the low-pass reconstruction.
        self.prev_f_lp = None
        self.prev_lam = LAM0.copy()             # APPLIED (EMA'd) lambda_{t-1}, for the pR_dot reconstruction
        self.lam_buf = init_lam_history()       # newest-first full-depth applied-lambda buffer (log-tap net input)
        self.lam_lp = LAM0.copy()               # in-loop EMA state (identical to deploy_prdot's continuity filter)
        self.lam_a = None if LAM_LP_TAU is None else self.dt / (LAM_LP_TAU + self.dt)

    def wrench_control(self, ep, eR, ev, ew, ang_vel):
        """Analytic PID -> desired 6-D load wrench w_d (kept from the classical law)."""
        self.intg_ep += ep * self.dt
        self.intg_eR += eR * self.dt
        f_L_d = (self.mass * self.g * self.e3
                 - self.Kp @ ep - self.Kv @ ev - self.Ki @ self.intg_ep)
        gyroscopic = np.cross(ang_vel, self.inertia @ ang_vel)
        tau_L_d = (gyroscopic
                   - self.KR @ eR - self.Kw @ ew - self.KiR @ self.intg_eR)
        return np.concatenate((f_L_d, tau_L_d))

    def prepare(self, pL, vL, R, omega, t, traj=None):
        """Phase 1 (per drone): wrench + reconstruct(pR_dot from own lambda history) +
        build the net input row. Stashes w_d / G+ / N for finalize. Returns the RAW
        (unnormalized) input row so the env can batch all N rows into ONE net forward.
        traj: this episode's reference (None -> default trajectory)."""
        ep, eR, ev, ew = error_calculation(pL, vL, R, omega, t, traj)
        w_d = self.wrench_control(ep, eR, ev, ew, omega)

        # Noise-robust pR_dot: low-pass the local base force from this drone's own view + lambda.
        vR, f_lp, G_pinv, Nmat = reconstruct_lp(R, vL, omega, w_d, self.prev_lam,
                                                self.prev_f_lp, self.Bb, self.L0,
                                                self.dt, self.recon_alpha)
        self._w_d, self._G_pinv, self._Nmat, self._f_lp = w_d, G_pinv, Nmat, f_lp
        self._vR = vR                                   # stash for diagnostics (own pR_dot)
        return build_input(t, vR, self.lam_buf)         # 66-D log-tap input (taps the full-depth buffer)

    def finalize(self, lam):
        """Phase 2 (per drone): EMA the net's lambda (identical to deploy_prdot's in-loop low-pass —
        the net was TRAINED against this filtered feedback), distribute forces from the APPLIED lambda
        REUSING the G+/N built in prepare (f = G+ w_d + N lambda, no recompute), and roll the local
        history (prev_lam + full-depth log-tap buffer). Returns the FULL (3n,) force; the caller keeps
        its own slice, and the env adds the residual delta_f AFTER this, in force space."""
        if self.lam_a is not None:                      # in-loop EMA (soft warm-start continuity)
            self.lam_lp = self.lam_a * lam + (1.0 - self.lam_a) * self.lam_lp
            lam = self.lam_lp
        f_full = self._G_pinv @ self._w_d + self._Nmat @ lam
        self.prev_f_lp = self._f_lp                     # roll filtered force
        self.prev_lam = lam.copy()                      # roll APPLIED lambda (recon feedback)
        self.lam_buf = push_lam(self.lam_buf, lam)      # roll APPLIED lambda into the log-tap history
        return f_full


class ResidualMARLEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "residual_marl_v0"}

    def __init__(
        self,
        fmu_filename="Base_Model.fmu",
        n_carriers=4,
        step_size=0.01,
        end_time=25.0,
        load_inertia=0.01,
        policy_path="il_actor_prdot_dagger_analytic.pt",   # frozen DAgger'd pR_dot policy each drone runs locally
        # --- residual RL (TWO-HEAD: action = [delta_lambda(n), delta_wrench(6)]) ---
        #   delta_lambda enters via N   -> nullspace -> reshapes drone trajectory, load-neutral.
        #   delta_wrench  enters via G+ -> range     -> trims the desired wrench, fixes load.
        #   Built in force space as [G+ @ dw + N @ dlam] so the two subspaces stay clean.
        cap_lam=0.4,       # cap on delta_lambda, fraction of ||lambda_base|| -> ~0.74N/drone nullspace
                           #   force (base null slice ~1.85N). Authority to reshape the loop; the blowup
                           #   guard (not this cap) prevents crashes, so this is set for AUTHORITY.
        cap_w=0.2,         # cap on delta_wrench, fraction of ||w_d||(~6.9) -> ~0.34N/drone load-trim force
                           #   (G+ gain ~0.25). Raised from 0.12 to give delta_wrench muscle to act on the
                           #   swing term below (a louder reward into a capped actuator does nothing).
        blowup_v=100.0,    # if any drone speed exceeds this, the state is diverging -> truncate (guard).
        blowup_penalty=100.0,   # one-shot penalty on a blowup-truncated step (raw, pre REWARD_SCALE).
        # --- reward weights (manifold-tracking; see expert_reference.py) ---
        manifold_w=1.0,    # distance to the expert loiter loop -> served by delta_lambda. Back to the PROVEN
                           #   value (regime-4 0.5 was marginal + muddied comparisons) so the delta_w ablation
                           #   is a clean single variable against the split1 baseline reward.
        stall_w=50.0,      # one-sided floor: penalize ||v|| < epsilon (fixed-wing STALL). Penalty =
                           #   d^2 + stall_lin*d, d=relu(eps+margin-v). 400 was OVERKILL: it drowned every
                           #   other term (jerk/swing became rounding errors) AND drove entropy collapse
                           #   (ent 4->-0.2, exploration died). The hinge+margin already give a firm bite
                           #   (~8.6 at a 0.15 dip) at w=50 -> stall stays enforced without steamrolling.
        stall_lin=1.0,     # linear-hinge coefficient (see stall_w). Raise to push the margin harder.
        stall_grace=20,    # skip stall penalty for the first N steps: drones accelerate from ~rest at reset,
                           #   an UNAVOIDABLE dip -> don't penalize it (lets stall_w be big without blowing up R).
        stall_margin=0.05, # penalize below (epsilon + margin), NOT just below epsilon -> the drones cruise
                           #   with a BUFFER above the true stall (0.25) and never graze it. Raise for more buffer.
        load_w=10.0,       # load-tracking. Back to moderate: it no longer has to bully one actuator —
                           #   delta_wrench is its dedicated knob, so tracking (delta_lambda) is conflict-free.
        swing_w=0.0,       # load VELOCITY-error ||ev||^2 damping. OFF (proven twice): swing is STRUCTURAL to
                           #   the residual's loop-correction, not a dampable side-effect -> penalizing it can't
                           #   reduce it and only craters vmin (drones slow to fake low load-velocity). The real
                           #   swing source is delta_wrench being (mis)used to fight stall -> leaks into load.
        jerk_w=8.0,        # per-drone velocity JERK ||v_t - 2 v_{t-1} + v_{t-2}||^2 -> punish JITTER. GRACED
                           #   at startup. CLIPPED per-step at jerk_cap. WHY CLIP: exploration inflates jerk
                           #   ~30x (det ~0.06 -> sampled ~2), so UNCLIPPED the cheapest way to cut the penalty
                           #   is to shrink log_std (kill exploration), NOT smooth the mean -> last run reward
                           #   improved via entropy collapse (4->-0.2) while deterministic jerk stayed flat.
                           #   Clipping zeros the gradient on the exploration spikes -> the ONLY way left to
                           #   lower the penalty is to smooth the sub-cap (structural) jitter of the MEAN.
        jerk_cap=0.05,     # per-step cap on jerk^2 (~||jerk||<0.22): above deterministic range (p95 0.014),
                           #   below exploration (~4) -> smooths the mean without the explore-less shortcut.
        coord_w=0.0,       # ||sum_i f_int||^2 COORDINATION: internal forces cancel iff drones agree; the
                           #   NON-cancellation is the nullspace leak that disturbs the load -> BOTH a
                           #   coordination and load-neutrality term (CLEAN swing fix, no airspeed cost).
                           #   Safe from all-zero degeneracy ONLY if manifold_w stays comparable. w=50 was a
                           #   DISASTER: policy leak is 0.79N (10x base) + exploration inflates it -> team_R
                           #   -800k, coord DOMINATED and steamrolled loop/load/swing (like stall_w=400).
                           #   w=3 = GENTLE retry: does coord drop WITHOUT wrecking tracking, and does swing
                           #   then fall? If swing still doesn't fall at gentle w -> leak isn't the swing
                           #   cause (maybe TORQUE leak / inherent desync) -> abandon coord for swing.
        expert_ref="expert_ref.npz",
        # --- classical config ---
        epsilon=0.25,
        phases=(0.0, np.pi / 2, 0.0, np.pi / 2),
        recon_tau=0.1,   # OUR estimate of the drone LLC time const, for reconstruct_lp (tunable; != plant)
        # --- sensing noise, per load channel (std; 0 => off) ---
        pos_noise=0.0, rot_noise=0.0, vel_noise=0.0, angvel_noise=0.0,
        noise_corr=0.0, own_noise=0.0, actuation_noise=0.0,
        # --- temporal desync (scalar or length-n) ---
        ctrl_delay=0, clock_offset=0.0,
        # --- ablation: zero the delta_wrench (range/load-trim) head -> delta_lambda-only policy.
        #     Tests whether the load head earns its keep or is just a load-disturbing stall crutch. ---
        disable_dw=False,
        # --- this episode's load-pose reference (callable t->(p,v,R,omega); None -> default trajectory).
        #     Set per-episode (env.traj = ...) for the trajectory library; kept on the INSTANCE (not a
        #     module global) so it's safe under multiprocessing / vectorized envs. ---
        traj=None,
    ):
        self.n = n_carriers
        self.dt = step_size
        self.end_time = end_time
        self.cap_lam = cap_lam
        self.cap_w = cap_w
        self.blowup_v = blowup_v
        self.blowup_penalty = blowup_penalty
        self.manifold_w, self.stall_w, self.load_w = manifold_w, stall_w, load_w
        self.swing_w = swing_w
        self.coord_w = coord_w
        self.jerk_w, self.jerk_cap = jerk_w, jerk_cap
        self.stall_lin, self.stall_grace, self.stall_margin = stall_lin, stall_grace, stall_margin
        self.expert_pos = np.load(expert_ref)["dpos"]   # (n, T, 3) per-drone reference loiter path
        self.epsilon = epsilon
        self.phases = np.asarray(phases, dtype=float)
        self.pos_noise = pos_noise
        self.rot_noise = rot_noise
        self.vel_noise = vel_noise
        self.angvel_noise = angvel_noise
        self.noise_corr = noise_corr
        self.own_noise = own_noise
        self.actuation_noise = actuation_noise
        self.ctrl_delay = self._broadcast(ctrl_delay).astype(int)
        self.clock_offset = self._broadcast(clock_offset).astype(float)
        self.disable_dw = disable_dw
        self.traj = traj                                  # per-episode reference (None -> default)

        self.possible_agents = [f"drone_{i}" for i in range(self.n)]
        self.agents = list(self.possible_agents)

        # Plant + constant parameters.
        self.plant = FMUPlantEnv(fmu_filename, n_carriers, step_size, end_time, load_inertia)
        self.plant.reset()
        vrs, fmu = self.plant.vrs, self.plant.fmu
        self.J = np.array(fmu.getReal([vrs[f"Load_Inertia_Matrix[1,{i}]"] for i in range(1, 10)])).reshape((3, 3), order="F")
        self.Bb = np.array(fmu.getReal([vrs[f"Attachment_Point_Vectors[1,{i}]"] for i in range(1, 13)])).reshape((self.n, 3))
        self.mass = fmu.getReal([vrs["Load_Mass"]])[0]
        self.L0 = fmu.getReal([vrs["Cable_Resting_Length"]])[0]

        # Frozen whole-vector policy, shared by all local replicas (no CasADi anywhere).
        # The net + normalization live on the env so the N per-drone rows can be run in
        # ONE batched forward; each LocalModelAgent owns only its geometry + history.
        ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
        hidden = tuple(ckpt.get("hidden", (128, 128)))        # match the F1 ckpt's capacity (256,256 for
        self.net = Actor(obs_dim=ckpt["obs_mean"].shape[1],   #   the log-tap net); pre-"hidden" ckpts = (128,128)
                         act_dim=self.n, hidden=hidden)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.obs_mean = ckpt["obs_mean"].astype(np.float32)   # (1, obs_dim)
        self.obs_std = ckpt["obs_std"].astype(np.float32)
        self.recon_alpha = step_size / (recon_tau + step_size)   # reconstruction filter (tunable estimate)
        self.locals = [
            LocalModelAgent(self.n, self.dt, self.phases, self.epsilon, self.L0,
                            self.mass, self.J, self.Bb, self.recon_alpha)
            for _ in range(self.n)
        ]

        # Actuator filter for the ACTUAL plant force.
        self.llc_alpha = step_size / (0.2 + step_size)
        self._Fz = self.mass * 9.81 / self.n

        # obs = load est(18) + own drone(6) + own f_g(3) + own f_lambda(3) + clock(14) = 44 ;
        # action = [delta_lambda(n), delta_wrench(6)].  clock features = loiter PHASE (zero-comms).
        self._clock_dim = clock_features(0.0).shape[0]
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(30 + self._clock_dim,), dtype=np.float32)
        self._act_space = spaces.Box(-1.0, 1.0, shape=(self.n + 6,), dtype=np.float32)

        self.np_random = np.random.default_rng()
        self.t = 0.0
        self._obs42 = None
        self._prev_f = None

    # ---- PettingZoo space accessors ----
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._act_space

    def state(self):
        """Global state for the CTDE centralized critic (training only)."""
        return self._obs42.astype(np.float32)

    # ---- helpers ----
    def _broadcast(self, val):
        arr = np.atleast_1d(np.asarray(val, dtype=float))
        if arr.size == 1:
            arr = np.full(self.n, arr.item())
        return arr

    @staticmethod
    def _clip_norm(v, max_norm):
        """Scale v so ||v|| <= max_norm (no-op if already inside or max_norm<=0)."""
        if max_norm <= 0:
            return v
        nrm = np.linalg.norm(v)
        return v * (max_norm / nrm) if nrm > max_norm else v

    def _unpack_load(self, obs42):
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)
        lin_vel = obs42[12:15]
        ang_vel = obs42[15:18]
        return pos, R, lin_vel, ang_vel

    @staticmethod
    def _project_SO3(M):
        U, _, Vt = np.linalg.svd(M)
        Rp = U @ Vt
        if np.linalg.det(Rp) < 0:
            U[:, -1] *= -1
            Rp = U @ Vt
        return Rp

    def _ar1(self, prev, sigma):
        if sigma <= 0:
            return prev
        eps = self.np_random.standard_normal(prev.shape)
        return self.noise_corr * prev + np.sqrt(1.0 - self.noise_corr ** 2) * sigma * eps

    def _update_estimates(self, obs42):
        """Each drone's sensed load estimate = per-agent delay + correlated noise,
        shared by that drone's local replica AND its observation."""
        pos, R, lin_vel, ang_vel = self._unpack_load(obs42)
        self._state_buffer.append((pos, R, lin_vel, ang_vel))

        self._noise_pos = self._ar1(self._noise_pos, self.pos_noise)
        self._noise_vel = self._ar1(self._noise_vel, self.vel_noise)
        self._noise_angvel = self._ar1(self._noise_angvel, self.angvel_noise)
        self._noise_rot = self._ar1(self._noise_rot, self.rot_noise)

        self._estimates = []
        for i in range(self.n):
            d = min(int(self.ctrl_delay[i]), len(self._state_buffer) - 1)
            p0, R0, v0, w0 = self._state_buffer[-1 - d]
            p = p0 + self._noise_pos[i]
            v = v0 + self._noise_vel[i]
            w = w0 + self._noise_angvel[i]
            Rn = self._project_SO3(R0 + self._noise_rot[i]) if self.rot_noise > 0 else R0
            self._estimates.append((p, Rn, v, w))

    def _build_obs(self, obs42):
        """Per-agent 24-D residual-policy observation: sensed load(18) + own drone(6)."""
        out = {}
        for i, name in enumerate(self.possible_agents):
            p, R, v, w = self._estimates[i]
            load18 = np.concatenate([p, R.flatten(order="C"), v, w])
            dpos = obs42[18 + 3 * i: 18 + 3 * i + 3]
            dvel = obs42[18 + 3 * self.n + 3 * i: 18 + 3 * self.n + 3 * i + 3]
            own = np.concatenate([dpos, dvel])
            if self.own_noise > 0:
                own = own + self.np_random.normal(0, self.own_noise, own.shape)
            clk = clock_features(self.t + self.clock_offset[i])   # loiter PHASE (per-drone clock)
            # + base force decomposition slices (one-step lag) + clock phase.
            out[name] = np.concatenate([load18, own, self._fg[i], self._flam[i], clk]).astype(np.float32)
        return out

    # ---- PettingZoo API ----
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        obs42, _ = self.plant.reset()
        for lm in self.locals:
            lm.reset()

        self.t = 0.0
        self._step = 0
        self._state_buffer = deque(maxlen=int(self.ctrl_delay.max()) + 1)
        self._noise_pos = np.zeros((self.n, 3))
        self._noise_vel = np.zeros((self.n, 3))
        self._noise_angvel = np.zeros((self.n, 3))
        self._noise_rot = np.zeros((self.n, 3, 3))
        self._obs42 = obs42
        self._prev_f = np.array([0.0, 0.0, self._Fz] * self.n)
        self._fg = np.zeros((self.n, 3))          # base load-serving force slice (obs, one-step lag)
        self._flam = np.zeros((self.n, 3))         # base nullspace force slice
        self._net_fint = np.zeros(3)               # sum of internal forces (coordination); set each step
        self._v_prev1 = None                       # drone velocities t-1, t-2 (for jerk = 2nd difference)
        self._v_prev2 = None
        self.agents = list(self.possible_agents)

        self._update_estimates(obs42)
        observations = self._build_obs(obs42)
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        # --- 1. Local replicas. Phase 1: each drone reconstructs pR_dot on its OWN load
        #        view + offset clock and builds its net input row. All N rows go through
        #        the net in ONE batched forward (independent inputs -> vectorized, NOT
        #        shared). Phase 2: each drone distributes forces from its own lambda vector
        #        (reusing phase-1 G+/N) and keeps its own slice. ---
        rows = []
        for i, lm in enumerate(self.locals):
            p_i, R_i, v_i, w_i = self._estimates[i]
            t_i = self.t + self.clock_offset[i]
            rows.append(lm.prepare(p_i, v_i, R_i, w_i, t_i, self.traj))
        Xn = ((np.stack(rows) - self.obs_mean) / self.obs_std).astype(np.float32)
        with torch.no_grad():
            lams = self.net(torch.tensor(Xn)).numpy()      # (n, n): row i = drone i's whole lambda vector

        f_base = np.zeros(3 * self.n)
        lam_own = np.zeros(self.n)
        for i, lm in enumerate(self.locals):
            f_full = lm.finalize(lams[i])
            sl = slice(3 * i, 3 * i + 3)
            f_base[sl] = f_full[sl]
            lam_own[i] = lams[i][i]
            self._fg[i] = (lm._G_pinv @ lm._w_d)[sl]      # own load-serving slice (for next obs)
            self._flam[i] = f_full[sl] - self._fg[i]      # own nullspace slice

        # --- 2. Residual RL (two-head): action = [delta_lambda(n), delta_wrench(6)]. Build the
        #        residual in force space as [G+ @ dw + N @ dlam] using THIS drone's own G+/N, so
        #        dlam lives in its nullspace (load-neutral) and dw in its range (load-correcting).
        #        f_eff = G+(w_d+dw) + N(lam+dlam) = f_base + [G+ dw + N dlam]; keep own slice. ---
        f_cmd = f_base.copy()
        delta_f = {}
        net_fint = np.zeros(3)               # sum of applied internal (nullspace) force slices -> COORDINATION
        for i, name in enumerate(self.possible_agents):
            a = np.asarray(actions[name], dtype=float)
            dlam = self._clip_norm(a[:self.n],        self.cap_lam * np.linalg.norm(lams[i]))
            dw   = self._clip_norm(a[self.n:self.n+6], self.cap_w   * np.linalg.norm(self.locals[i]._w_d))
            if self.disable_dw:                       # delta_lambda-only ablation: no load-space trim
                dw = np.zeros_like(dw)
            nulls = self.locals[i]._Nmat @ dlam                       # residual nullspace force (3n,)
            df_full = self.locals[i]._G_pinv @ dw + nulls            # (3n,) range + null
            df = df_full[3 * i: 3 * i + 3]
            f_cmd[3 * i: 3 * i + 3] = f_base[3 * i: 3 * i + 3] + df
            net_fint += self._flam[i] + nulls[3 * i: 3 * i + 3]      # total applied internal force, this drone
            delta_f[name] = df
        self._net_fint = net_fint            # =0 iff internal forces cancel (coordinated); leak disturbs load

        # --- 3. Actuation noise (optional). ---
        if self.actuation_noise > 0:
            f_cmd = f_cmd + self.np_random.normal(0, self.actuation_noise, f_cmd.shape)

        # --- 4. LLC filter -> derivatives -> step the plant. ---
        ff = self.llc_alpha * f_cmd + (1 - self.llc_alpha) * self._prev_f
        deriv = (ff - self._prev_f) / self.dt
        self._prev_f = ff.copy()
        obs42, _, _, truncated, _ = self.plant.step(np.concatenate([ff, deriv]))
        self.t += self.dt
        self._step += 1

        # --- 4b. Blowup guard: a bad exploration action can collapse tension -> velocity
        #        explodes -> NaN state -> the NEXT step's reconstruct SVD throws and kills the
        #        whole run. Catch it here: truncate cleanly with a penalty (no crash), which also
        #        teaches the policy to avoid it. Return BEFORE anything touches the bad state. ---
        finite = np.isfinite(obs42).all()
        vmax = np.max(np.abs(obs42[18 + 3 * self.n: 18 + 6 * self.n])) if finite else np.inf
        if (not finite) or vmax > self.blowup_v:
            self.agents = []
            zeros = {a: np.zeros(self._obs_space.shape[0], np.float32) for a in self.possible_agents}
            pen = {a: -self.blowup_penalty for a in self.possible_agents}
            term = {a: False for a in self.possible_agents}
            trunc = {a: True for a in self.possible_agents}
            info = {a: {"lambda": 0.0, "prdot_own": 0.0, "loop_dist": 5.0,
                        "load_err": 5.0, "load_verr": 5.0, "min_speed": 0.0, "coord": 5.0, "jerk": 5.0,
                        "blowup": True}
                    for a in self.possible_agents}
            return zeros, pen, term, trunc, info

        self._obs42 = obs42

        # --- 5. Reward (TRUE state): TIME-INDEXED tracking of the expert loiter + don't stall
        #        + load guardrail. Track term = squared distance to the PHASE-CORRECT expert
        #        point p_i^central(t) (the clock in obs makes this achievable); stall = floor. ---
        npos, nR, nvel, nw = self._unpack_load(obs42)
        ep, _, ev, _ = error_calculation(npos, nvel, nR, nw, self.t, self.traj)
        load_err2 = float(ep @ ep)
        swing2 = float(ev @ ev)          # load velocity-error -> damping/smoothness signal (global)
        coord2 = float(self._net_fint @ self._net_fint)   # ||sum internal force||^2 -> coordination/leak (global)
        idx = min(self._step, self.expert_pos.shape[1] - 1)   # current step -> phase-correct expert point
        v_cur = obs42[18 + 3 * self.n: 18 + 6 * self.n].reshape(self.n, 3)   # drone velocity vectors
        # per-drone JERK = ||v_t - 2 v_{t-1} + v_{t-2}||^2 (needs 2 steps of history + past startup grace)
        have_jerk = self._v_prev1 is not None and self._v_prev2 is not None and self._step > self.stall_grace
        jerk_vec = (v_cur - 2 * self._v_prev1 + self._v_prev2) if have_jerk else np.zeros((self.n, 3))
        rewards, loop_d = {}, {}
        min_speed = np.inf                # slowest drone this step -> stall monitor (DET_stall)
        for i, name in enumerate(self.possible_agents):
            p_i = obs42[18 + 3 * i: 18 + 3 * i + 3]
            v_i = float(np.linalg.norm(v_cur[i]))
            min_speed = min(min_speed, v_i)
            d2 = float(np.sum((self.expert_pos[i][idx] - p_i) ** 2))   # TIME-INDEXED target
            d_stall = max(0.0, (self.epsilon + self.stall_margin) - v_i)   # depth below CRUISE floor (eps+margin)
            stall = 0.0 if self._step <= self.stall_grace else d_stall ** 2 + self.stall_lin * d_stall
            jerk2 = min(float(jerk_vec[i] @ jerk_vec[i]), self.jerk_cap)   # per-drone jerk, CLIPPED (see jerk_cap)
            rewards[name] = -(self.manifold_w * d2 + self.stall_w * stall
                              + self.load_w * load_err2 + self.swing_w * swing2
                              + self.coord_w * coord2 + self.jerk_w * jerk2)
            loop_d[name] = d2 ** 0.5
        self._v_prev2 = self._v_prev1        # roll velocity history for next step's jerk
        self._v_prev1 = v_cur

        # --- 6. Sense the new state and package outputs. ---
        self._update_estimates(obs42)
        observations = self._build_obs(obs42)
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: truncated for a in self.possible_agents}
        load_err = load_err2 ** 0.5
        load_verr = swing2 ** 0.5          # load velocity-error norm = the SWING rate (for eval/demo)
        coord = coord2 ** 0.5              # ||sum internal force|| (for calibrating coord_w + monitoring)
        jerk = float(np.mean(np.linalg.norm(jerk_vec, axis=1)))   # mean per-drone velocity jerk (JITTER monitor)
        infos = {name: {"lambda": lam_own[i],
                        "prdot_own": float(np.linalg.norm(self.locals[i]._vR[i])),
                        "loop_dist": loop_d[name],
                        "load_err": load_err,
                        "load_verr": load_verr,
                        "min_speed": min_speed,
                        "coord": coord,
                        "jerk": jerk}
                 for i, name in enumerate(self.possible_agents)}

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.plant.close()
