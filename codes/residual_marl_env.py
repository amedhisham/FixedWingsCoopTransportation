"""
residual_marl_env.py
====================
Formulation-2 (residual RL) multi-agent environment — NETWORK-based.

Each drone no longer runs a CasADi optimizer. Instead it runs a LocalModelAgent:
a fully-local replica of the whole system built from the frozen pR_dot policy plus a
kinematic reconstruction of the carriers. Per step, for each drone i (on its OWN,
possibly desynced, load view and offset clock):

    f_base_i  = slice_i( LocalModelAgent_i.compute_forces( its own load view ) )
    delta_f_i = action_i * residual_scale            # residual RL (optionally capped)
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
from collect_prdot_data import reconstruct, build_input, LAM0


class LocalModelAgent:
    """One drone's local replica of the pR_dot WHOLE-VECTOR policy. Fully local, zero
    communication. Split into two phases so the env can batch the N net forwards:
      prepare()  -> wrench + reconstruct(all N pR_dot from its own load estimate + its own
                    lambda history) + build the net input row (env runs the net).
      finalize() -> distribute forces from its whole lambda vector (reusing prepare's G+/N),
                    keep its own slice, roll history.
    STATEFUL: carries its own lambda / G+ / N / w_d history for the analytic reconstruction
    — exactly deploy_prdot's single loop, one copy per drone."""

    def __init__(self, n, dt, phases, epsilon, L0, mass, J, Bb):
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
        self.reset()

    def reset(self):
        self.intg_ep = np.zeros(3)
        self.intg_eR = np.zeros(3)
        # Local lambda / derivative history for the analytic reconstruction.
        self.prev_G_pinv = None
        self.prev_Nmat = None
        self.prev_w_d = None
        self.prev_lam = LAM0.copy()
        self.prev_prev_lam = LAM0.copy()

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

    def prepare(self, pL, vL, R, omega, t):
        """Phase 1 (per drone): wrench + reconstruct(pR_dot from own lambda history) +
        build the net input row. Stashes w_d / G+ / N for finalize. Returns the RAW
        (unnormalized) input row so the env can batch all N rows into ONE net forward."""
        ep, eR, ev, ew = error_calculation(pL, vL, R, omega, t)
        w_d = self.wrench_control(ep, eR, ev, ew, omega)

        # Optimizer-exact pR_dot from this drone's OWN previous lambda + load estimate.
        lamdot_prev = (self.prev_lam - self.prev_prev_lam) / self.dt
        vR, G_pinv, Nmat = reconstruct(R, vL, omega, w_d, self.prev_lam, lamdot_prev,
                                       self.prev_G_pinv, self.prev_Nmat, self.prev_w_d,
                                       self.Bb, self.L0, self.dt)
        self._w_d, self._G_pinv, self._Nmat = w_d, G_pinv, Nmat
        return build_input(t, vR, self.prev_lam)

    def finalize(self, lam):
        """Phase 2 (per drone): distribute forces from THIS drone's whole lambda vector,
        REUSING the G+/N built in prepare (f = G+ w_d + N lambda, no recompute). Rolls the
        local history forward. Returns the FULL (3n,) force; caller keeps its own slice."""
        f_full = self._G_pinv @ self._w_d + self._Nmat @ lam
        self.prev_G_pinv, self.prev_Nmat, self.prev_w_d = self._G_pinv, self._Nmat, self._w_d
        self.prev_prev_lam, self.prev_lam = self.prev_lam, lam.copy()
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
        policy_path="il_actor_prdot_dagger.pt",   # frozen DAgger'd pR_dot policy each drone runs locally
        # --- residual RL ---
        residual_scale=1.0,
        residual_cap=0.5,
        # --- reward weights ---
        w_p=1.0, w_R=1.0, w_df=0.01,
        # --- classical config ---
        epsilon=0.25,
        phases=(0.0, np.pi / 2, 0.0, np.pi / 2),
        # --- sensing noise, per load channel (std; 0 => off) ---
        pos_noise=0.0, rot_noise=0.0, vel_noise=0.0, angvel_noise=0.0,
        noise_corr=0.0, own_noise=0.0, actuation_noise=0.0,
        # --- temporal desync (scalar or length-n) ---
        ctrl_delay=0, clock_offset=0.0,
    ):
        self.n = n_carriers
        self.dt = step_size
        self.end_time = end_time
        self.residual_scale = residual_scale
        self.residual_cap = residual_cap
        self.w_p, self.w_R, self.w_df = w_p, w_R, w_df
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
        self.net = Actor(obs_dim=ckpt["obs_mean"].shape[1], act_dim=self.n)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.obs_mean = ckpt["obs_mean"].astype(np.float32)   # (1, obs_dim)
        self.obs_std = ckpt["obs_std"].astype(np.float32)
        self.locals = [
            LocalModelAgent(self.n, self.dt, self.phases, self.epsilon, self.L0,
                            self.mass, self.J, self.Bb)
            for _ in range(self.n)
        ]

        # Actuator filter for the ACTUAL plant force.
        self.llc_alpha = step_size / (0.2 + step_size)
        self._Fz = self.mass * 9.81 / self.n

        # obs = load estimate(18) + own drone state(6) = 24 ; action = delta_f(3).
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)
        self._act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

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
            out[name] = np.concatenate([load18, own]).astype(np.float32)
        return out

    # ---- PettingZoo API ----
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        obs42, _ = self.plant.reset()
        for lm in self.locals:
            lm.reset()

        self.t = 0.0
        self._state_buffer = deque(maxlen=int(self.ctrl_delay.max()) + 1)
        self._noise_pos = np.zeros((self.n, 3))
        self._noise_vel = np.zeros((self.n, 3))
        self._noise_angvel = np.zeros((self.n, 3))
        self._noise_rot = np.zeros((self.n, 3, 3))
        self._obs42 = obs42
        self._prev_f = np.array([0.0, 0.0, self._Fz] * self.n)
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
            rows.append(lm.prepare(p_i, v_i, R_i, w_i, t_i))
        Xn = ((np.stack(rows) - self.obs_mean) / self.obs_std).astype(np.float32)
        with torch.no_grad():
            lams = self.net(torch.tensor(Xn)).numpy()      # (n, n): row i = drone i's whole lambda vector

        f_base = np.zeros(3 * self.n)
        lam_own = np.zeros(self.n)
        for i, lm in enumerate(self.locals):
            f_full = lm.finalize(lams[i])
            f_base[3 * i: 3 * i + 3] = f_full[3 * i: 3 * i + 3]
            lam_own[i] = lams[i][i]

        # --- 2. Residual RL: add the (norm-capped) per-agent action. ---
        f_cmd = f_base.copy()
        delta_f = {}
        for i, name in enumerate(self.possible_agents):
            base_i = f_base[3 * i: 3 * i + 3]
            df = np.asarray(actions[name], dtype=float) * self.residual_scale
            if self.residual_cap:
                max_res = self.residual_cap * np.linalg.norm(base_i)
                nrm = np.linalg.norm(df)
                if nrm > max_res and nrm > 0:
                    df *= max_res / nrm
            f_cmd[3 * i: 3 * i + 3] = base_i + df
            delta_f[name] = df

        # --- 3. Actuation noise (optional). ---
        if self.actuation_noise > 0:
            f_cmd = f_cmd + self.np_random.normal(0, self.actuation_noise, f_cmd.shape)

        # --- 4. LLC filter -> derivatives -> step the plant. ---
        ff = self.llc_alpha * f_cmd + (1 - self.llc_alpha) * self._prev_f
        deriv = (ff - self._prev_f) / self.dt
        self._prev_f = ff.copy()
        obs42, _, _, truncated, _ = self.plant.step(np.concatenate([ff, deriv]))
        self.t += self.dt
        self._obs42 = obs42

        # --- 5. Reward from the TRUE (noise-free) new load state. ---
        npos, nR, nvel, nw = self._unpack_load(obs42)
        ep, eR, _, _ = error_calculation(npos, nvel, nR, nw, self.t)
        track = self.w_p * float(ep @ ep) + self.w_R * float(eR @ eR)
        rewards = {name: -track - self.w_df * float(delta_f[name] @ delta_f[name])
                   for name in self.possible_agents}

        # --- 6. Sense the new state and package outputs. ---
        self._update_estimates(obs42)
        observations = self._build_obs(obs42)
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: truncated for a in self.possible_agents}
        infos = {name: {"lambda": lam_own[i]} for i, name in enumerate(self.possible_agents)}

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.plant.close()
