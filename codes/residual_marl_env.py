"""
residual_marl_env.py
====================
Formulation-2 (residual RL) multi-agent environment.

It wraps the validated FMUPlantEnv (the plant) and four ClassicalAgents (the
per-drone controller+optimizer "expert"), and exposes a PettingZoo ParallelEnv
so a MAPPO trainer can later plug in. Per step, for each drone i:

    f_base_i  = slice_i( ClassicalAgent_i.compute_forces( its own load-state view ) )   # the stitch
    delta_f_i = action_i * residual_scale        # fixed scale -> constant physical meaning
    f_cmd_i   = f_base_i + delta_f_i             # residual RL (delta_f optionally safety-clipped)
    -> LLC filter -> plant.step()

All noise sources default to 0. With zero noise AND zero residual actions, the
stitch collapses to the single coherent controller, so the env reproduces the
main_env baseline exactly. That is the validation target before any learning.

Nothing here trains anything. The `actions` handed to step() come from a plain
script now (zeros), and from a neural-network policy later.
"""

import functools
from collections import deque
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from fmu_plant_env import FMUPlantEnv
from classical_agent import ClassicalAgent
from controller import error_calculation
from optimizer import init_optimizer


class ResidualMARLEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "residual_marl_v0"}

    def __init__(
        self,
        fmu_filename="Base_Model.fmu",
        n_carriers=4,
        step_size=0.01,
        end_time=25.0,
        load_inertia=0.01,
        # --- residual RL ---
        residual_scale=1.0,        # action in [-1,1] -> delta_f = action * residual_scale  (Newtons)
        residual_cap=0.5,          # optional safety clip: ||delta_f|| <= residual_cap*||f_base|| (0/None = off)
        # --- reward weights ---
        w_p=1.0, w_R=1.0, w_df=0.01,
        # --- classical/optimizer config ---
        epsilon=0.25,
        phases=(0.0, np.pi / 2, 0.0, np.pi / 2),
        # --- sensing noise, per load channel (std of zero-mean Gaussian; 0 => off) ---
        # These corrupt each drone's load-state estimate, shared by expert AND policy obs.
        pos_noise=0.0,             # std on load position          (m)
        rot_noise=0.0,             # std on load orientation entries (unitless; re-projected to SO(3))
        vel_noise=0.0,             # std on load linear velocity   (m/s)
        angvel_noise=0.0,          # std on load angular velocity  (rad/s)
        noise_corr=0.0,            # temporal correlation of sensing noise: 0=white, ->1=smooth (AR(1) rho)
        own_noise=0.0,             # std on a drone's own pos/vel in its observation (policy only)
        actuation_noise=0.0,       # noise on the commanded forces
        # --- temporal desync (scalar => same for all, or length-n list) ---
        ctrl_delay=0,              # integer steps of delay on the load estimate each drone senses
        clock_offset=0.0,          # seconds added to each expert's clock (reference + sinusoid; expert only)
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

        # ---- PettingZoo agent bookkeeping ----
        self.possible_agents = [f"drone_{i}" for i in range(self.n)]
        self.agents = list(self.possible_agents)

        # ---- Build the plant and read its constant parameters ----
        self.plant = FMUPlantEnv(fmu_filename, n_carriers, step_size, end_time, load_inertia)
        self.plant.reset()  # initialize the FMU so parameters are readable
        vrs, fmu = self.plant.vrs, self.plant.fmu
        self.J = np.array(fmu.getReal([vrs[f"Load_Inertia_Matrix[1,{i}]"] for i in range(1, 10)])).reshape((3, 3), order="F")
        self.Bb = np.array(fmu.getReal([vrs[f"Attachment_Point_Vectors[1,{i}]"] for i in range(1, 13)])).reshape((self.n, 3))
        self.mass = fmu.getReal([vrs["Load_Mass"]])[0]
        self.L0 = fmu.getReal([vrs["Cable_Resting_Length"]])[0]

        # ---- One shared CasADi solver + four independent ClassicalAgents ----
        # The solver is stateless (all history passed as params), so sharing is safe
        # and avoids compiling IPOPT four times.
        self.solver = init_optimizer(self.L0, self.n, 0.0, 0.0, self.phases)
        self.experts = [
            ClassicalAgent(self.n, self.dt, self.phases, self.epsilon, self.L0,
                           self.mass, self.J, self.Bb, solver=self.solver)
            for _ in range(self.n)
        ]

        # ---- Low-level actuator filter (finite bandwidth), same as main_env ----
        self.llc_alpha = step_size / (0.2 + step_size)
        self._Fz = 0.7 * 9.81 / 4  # hover force per drone, used to seed the filter

        # ---- Spaces ----
        # obs = load state(18) + own drone state(6) = 24 ; action = delta_f(3), normalized
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)
        self._act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.np_random = np.random.default_rng()
        self.t = 0.0
        self._obs42 = None
        self._prev_f = None

    # ------------------------------------------------------------------ #
    #  PettingZoo required space accessors                                #
    # ------------------------------------------------------------------ #
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._act_space

    def state(self):
        """Global state for the CTDE centralized critic (training only)."""
        return self._obs42.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _broadcast(self, val):
        """Turn a scalar or length-n value into a length-n numpy array."""
        arr = np.atleast_1d(np.asarray(val, dtype=float))
        if arr.size == 1:
            arr = np.full(self.n, arr.item())
        return arr

    def _unpack_load(self, obs42):
        """True load state from the 42-D plant observation."""
        pos = obs42[0:3]
        R = np.round(obs42[3:12].reshape((3, 3), order="C"), 6)  # round like main_env
        lin_vel = obs42[12:15]
        ang_vel = obs42[15:18]
        return pos, R, lin_vel, ang_vel

    @staticmethod
    def _project_SO3(M):
        """Nearest rotation matrix (keeps a noised R a valid orientation)."""
        U, _, Vt = np.linalg.svd(M)
        Rp = U @ Vt
        if np.linalg.det(Rp) < 0:
            U[:, -1] *= -1
            Rp = U @ Vt
        return Rp

    def _ar1(self, prev, sigma):
        """Advance one AR(1) noise channel: n <- rho*n + sqrt(1-rho^2)*sigma*eps.
        rho = noise_corr. rho=0 -> white noise (std sigma each step); rho->1 ->
        temporally-smooth noise with the same stationary std sigma."""
        if sigma <= 0:
            return prev                        # stays zero -> baseline untouched
        eps = self.np_random.standard_normal(prev.shape)
        return self.noise_corr * prev + np.sqrt(1.0 - self.noise_corr ** 2) * sigma * eps

    def _update_estimates(self, obs42):
        """Compute each drone's SENSED load estimate = per-agent delay + shared,
        temporally-correlated load noise. Stored once and used by BOTH the expert
        and the policy observation (a drone has one load sensor feeding both)."""
        pos, R, lin_vel, ang_vel = self._unpack_load(obs42)
        self._state_buffer.append((pos, R, lin_vel, ang_vel))   # newest true state

        # Advance the per-drone, per-channel correlated noise one step.
        self._noise_pos = self._ar1(self._noise_pos, self.pos_noise)
        self._noise_vel = self._ar1(self._noise_vel, self.vel_noise)
        self._noise_angvel = self._ar1(self._noise_angvel, self.angvel_noise)
        self._noise_rot = self._ar1(self._noise_rot, self.rot_noise)

        self._estimates = []
        for i in range(self.n):
            d = min(int(self.ctrl_delay[i]), len(self._state_buffer) - 1)
            p0, R0, v0, w0 = self._state_buffer[-1 - d]          # delay: state d steps ago
            p = p0 + self._noise_pos[i]
            v = v0 + self._noise_vel[i]
            w = w0 + self._noise_angvel[i]
            Rn = self._project_SO3(R0 + self._noise_rot[i]) if self.rot_noise > 0 else R0
            self._estimates.append((p, Rn, v, w))

    def _build_obs(self, obs42):
        """Per-agent 24-D observation: sensed load estimate(18) + own drone state(6).
        The load part is the SAME estimate the expert used (one sensor per drone)."""
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

    # ------------------------------------------------------------------ #
    #  PettingZoo API                                                     #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        obs42, _ = self.plant.reset()
        for ex in self.experts:
            ex.reset()

        self.t = 0.0
        # Ring buffer of past TRUE load states, for per-agent delay.
        self._state_buffer = deque(maxlen=int(self.ctrl_delay.max()) + 1)
        # Per-drone, per-channel correlated sensing-noise state (starts at 0).
        self._noise_pos = np.zeros((self.n, 3))
        self._noise_vel = np.zeros((self.n, 3))
        self._noise_angvel = np.zeros((self.n, 3))
        self._noise_rot = np.zeros((self.n, 3, 3))
        self._obs42 = obs42
        self._prev_f = np.array([0.0, 0.0, self._Fz] * self.n)
        self.agents = list(self.possible_agents)

        self._update_estimates(obs42)          # sense the initial state
        observations = self._build_obs(obs42)
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        # --- 1. The stitch: each expert runs on its OWN sensed load estimate
        #        (already delayed + noised, shared with that drone's observation),
        #        on its own offset clock, and we keep only its own slice ---
        f_base = np.zeros(3 * self.n)
        for i, ex in enumerate(self.experts):
            p_i, R_i, v_i, w_i = self._estimates[i]
            t_i = self.t + self.clock_offset[i]                  # clock desync (expert only)
            f_full, _ = ex.compute_forces(p_i, v_i, R_i, w_i, t_i)
            f_base[3 * i: 3 * i + 3] = f_full[3 * i: 3 * i + 3]

        # --- 2. Residual RL: add the (norm-capped) per-agent action ---
        f_cmd = f_base.copy()
        delta_f = {}
        for i, name in enumerate(self.possible_agents):
            base_i = f_base[3 * i: 3 * i + 3]
            # action in [-1,1]^3 -> residual in Newtons via a FIXED scale (constant meaning).
            df = np.asarray(actions[name], dtype=float) * self.residual_scale
            # optional safety clip: keep ||df|| <= residual_cap * ||f_base_i|| (0/None disables).
            if self.residual_cap:
                max_res = self.residual_cap * np.linalg.norm(base_i)
                nrm = np.linalg.norm(df)
                if nrm > max_res and nrm > 0:
                    df *= max_res / nrm
            f_cmd[3 * i: 3 * i + 3] = base_i + df
            delta_f[name] = df

        # --- 3. Actuation noise (optional) ---
        if self.actuation_noise > 0:
            f_cmd = f_cmd + self.np_random.normal(0, self.actuation_noise, f_cmd.shape)

        # --- 4. LLC filter -> derivatives -> step the plant ---
        ff = self.llc_alpha * f_cmd + (1 - self.llc_alpha) * self._prev_f
        deriv = (ff - self._prev_f) / self.dt
        self._prev_f = ff.copy()
        obs42, _, _, truncated, _ = self.plant.step(np.concatenate([ff, deriv]))
        self.t += self.dt
        self._obs42 = obs42

        # --- 5. Reward from the TRUE (noise-free) new load state ---
        npos, nR, nvel, nw = self._unpack_load(obs42)
        ep, eR, _, _ = error_calculation(npos, nvel, nR, nw, self.t)
        track = self.w_p * float(ep @ ep) + self.w_R * float(eR @ eR)
        rewards = {name: -track - self.w_df * float(delta_f[name] @ delta_f[name])
                   for name in self.possible_agents}

        # --- 6. Sense the new state (shared expert/obs estimate) and package outputs ---
        self._update_estimates(obs42)
        observations = self._build_obs(obs42)
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: truncated for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.plant.close()
