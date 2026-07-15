"""
fmu_plant_env.py
================
The "dumb core" environment: a 1:1 Gymnasium wrapper around the Base_Model FMU.

This layer knows NOTHING about the classical controller, the optimizer, the
residual RL action, or the reward. Its only job is:

    raw forces (+ derivatives)  --->  [ FMU physics ]  --->  raw states

Everything intelligent (4x controller+optimizer, disturbances, residual, reward,
per-agent observation slicing) is built ON TOP of this in a separate wrapper.
Keeping this core faithful to the FMU's real input/output ports is what lets us
reuse it unchanged across Formulation 1, Formulation 2, and pure-classical runs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import fmpy
from fmpy.fmi2 import FMU2Slave


class FMUPlantEnv(gym.Env):
    """A minimal Gymnasium environment that steps the Base_Model FMU.

    Observation (42,) : the raw payload + drone state, in this fixed order
        [ load_pos(3), load_orientation(9), load_linvel(3),
          load_angvel(3), drone_pos(12), drone_vel(12) ]

    Action (24,)      : the raw FMU inputs, in this fixed order
        [ cable_forces(12), cable_force_derivatives(12) ]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        fmu_filename: str = "Base_Model.fmu",
        n_carriers: int = 4,
        step_size: float = 0.01,     # 100 Hz control loop, same as main.py
        end_time: float = 25.0,      # episode length in seconds
        load_inertia: float = 0.01,  # scalar -> J_L = load_inertia * I3
    ):
        super().__init__()

        self.fmu_filename = fmu_filename
        self.n_carriers = n_carriers
        self.step_size = step_size
        self.end_time = end_time
        self.load_inertia = load_inertia

        self.n_forces = 3 * n_carriers   # 12 : three force components per drone
        self.time = 0.0
        self._instantiated = False       # becomes True after first FMU setup

        # ---- 1. Parse the FMU's variable dictionary (name -> value reference) ----
        # A "value reference" is the integer address the FMI standard uses to
        # read/write a variable. We look them up once, by name, and cache them.
        self.model_desc = fmpy.read_model_description(fmu_filename)
        self.vrs = {v.name: v.valueReference for v in self.model_desc.modelVariables}

        # ---- 2. Pre-compute the ordered value-reference lists we will reuse ----
        # Inputs we WRITE every step:
        self.force_vrs = [self.vrs[f"Desired_Cable_Forces[{i}]"]
                          for i in range(1, self.n_forces + 1)]
        self.deriv_vrs = [self.vrs[f"Desired_Cable_Forces_Derivatives[{i}]"]
                          for i in range(1, self.n_forces + 1)]

        # Outputs we READ every step, concatenated into ONE list so a single
        # getReal() call returns the whole 42-D observation in the right order.
        load_pos_vrs    = [self.vrs[f"Load_Position[{i}]"]     for i in range(1, 4)]
        load_orient_vrs = [self.vrs[f"Load_Orientation[{r},{c}]"]
                           for r in range(1, 4) for c in range(1, 4)]  # row-major
        load_linvel_vrs = [self.vrs[f"Load_LinVelocity[{i}]"]  for i in range(1, 4)]
        load_angvel_vrs = [self.vrs[f"Load_AngVelocity[{i},1]"] for i in range(1, 4)]
        drone_pos_vrs   = [self.vrs[f"Drone_Positions[{i}]"]   for i in range(1, self.n_forces + 1)]
        drone_vel_vrs   = [self.vrs[f"Drones_LinVelocity[{i}]"] for i in range(1, self.n_forces + 1)]

        self.obs_vrs = (load_pos_vrs + load_orient_vrs + load_linvel_vrs
                        + load_angvel_vrs + drone_pos_vrs + drone_vel_vrs)
        self.obs_dim = len(self.obs_vrs)   # == 42

        # Parameters we SET once at every reset (matches main.py: only inertia).
        self.inertia_vrs = [self.vrs[f"Load_Inertia_Matrix[1,{i}]"] for i in range(1, 10)]

        # ---- 3. Declare the Gymnasium spaces ----
        # Observation: unbounded box (positions/velocities have no natural bound).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        # Action: 24 raw FMU inputs. Bounds here are nominal/generous - this dumb
        # core does not enforce task limits (the wrapper caps the residual).
        act_high = np.concatenate([
            np.full(self.n_forces, 100.0),    # forces      (N)
            np.full(self.n_forces, 1.0e4),    # derivatives (N/s)
        ]).astype(np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        # ---- 4. Extract and instantiate the FMU binary (done once) ----
        self.unzipdir = fmpy.extract(fmu_filename)
        self.fmu = FMU2Slave(
            guid=self.model_desc.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.model_desc.coSimulation.modelIdentifier,
            instanceName="PayloadSim",
        )
        self.fmu.instantiate()

    # ------------------------------------------------------------------ #
    #  FMU lifecycle helpers                                             #
    # ------------------------------------------------------------------ #
    def _enter_init_and_set_params(self):
        """Run the FMI initialization handshake and inject parameters."""
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()

        # Inject the load inertia matrix (flattened column-major, as Simulink expects).
        J_L = self.load_inertia * np.eye(3)
        self.fmu.setReal(self.inertia_vrs, J_L.flatten("F").tolist())

        self.fmu.exitInitializationMode()

    def _read_observation(self) -> np.ndarray:
        """Read all 42 state outputs from the FMU in one call."""
        return np.array(self.fmu.getReal(self.obs_vrs), dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                     #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds gym's RNG (self.np_random)

        # If the FMU was already initialized in a previous episode, roll it back
        # to the freshly-instantiated state before re-initializing.
        if self._instantiated:
            self.fmu.reset()
        self._instantiated = True

        self._enter_init_and_set_params()
        self.time = 0.0

        obs = self._read_observation()
        info = {"time": self.time}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).ravel()

        # Split the 24-D action into the two FMU input ports.
        forces      = action[: self.n_forces].tolist()
        derivatives = action[self.n_forces:].tolist()

        # A. Write the inputs into the FMU.
        self.fmu.setReal(self.force_vrs, forces)
        self.fmu.setReal(self.deriv_vrs, derivatives)

        # B. Advance the physics by one control step.
        self.fmu.doStep(currentCommunicationPoint=self.time,
                        communicationStepSize=self.step_size)
        self.time += self.step_size

        # C. Read the resulting state.
        obs = self._read_observation()

        # D. Bookkeeping. The dumb plant has no task reward and no failure
        #    condition; the wrapper defines those. We only truncate on time.
        reward = 0.0
        terminated = False
        truncated = self.time >= self.end_time - 1e-9
        info = {"time": self.time}

        return obs, reward, terminated, truncated, info

    def close(self):
        if getattr(self, "fmu", None) is not None:
            try:
                self.fmu.terminate()
            finally:
                self.fmu.freeInstance()
                self.fmu = None
