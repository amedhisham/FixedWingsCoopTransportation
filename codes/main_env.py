"""
main_env.py
===========
Validation harness: run the EXISTING classical controller + optimizer, but drive
the FMU through FMUPlantEnv instead of raw fmpy calls. If the plots match main.py,
the env is a faithful drop-in replacement for the hand-written FMU loop.

Nothing about RL here yet. This only swaps the plumbing:
    main.py   :  fmu.getReal(...) / fmu.setReal(...) / fmu.doStep(...)
    main_env  :  env.reset() / env.step(action)
The control math (error -> wrench -> optimizer -> cable forces -> LLC) is identical.
"""

import numpy as np
import matplotlib.pyplot as plt
import time as time_module

from fmu_plant_env import FMUPlantEnv
from controller import error_calculation, wrench_controller
from optimizer import cable_force_calculation, init_optimizer, optimizer


# ------------------------------------------------------------------ #
#  Helper: unpack the flat 42-D observation into named states         #
# ------------------------------------------------------------------ #
def unpack_obs(obs, n_carriers):
    """Slice the observation into the pieces the controller/optimizer expect.

    Layout (see FMUPlantEnv):
      [ load_pos(3), load_orient(9), load_linvel(3),
        load_angvel(3), drone_pos(3n), drone_vel(3n) ]
    """
    load_pos    = obs[0:3]
    # Row-major 9 -> 3x3 (order='C'), matching main.py's reshape + rounding.
    load_orient = np.round(obs[3:12].reshape((3, 3), order="C"), decimals=6)
    load_linvel = obs[12:15]
    load_angvel = obs[15:18]
    drone_pos   = obs[18:18 + 3 * n_carriers].reshape((n_carriers, 3))
    drone_vel   = obs[18 + 3 * n_carriers:].reshape((n_carriers, 3))
    return load_pos, load_orient, load_linvel, load_angvel, drone_pos, drone_vel


# ------------------------------------------------------------------ #
#  Configuration (mirrors main.py)                                    #
# ------------------------------------------------------------------ #
n_carriers = 4
epsilon = 0.25
w_pos, w_vel = 0, 0
phases = np.array([0, np.pi / 2, 0, np.pi / 2])
bypass_optimizer = 0
bypass_controller = 0

step_size = 0.01
end_time = 25.0

# Low-level controller filter (finite actuator bandwidth), same as main.py.
llc_tau = 0.2
llc_alpha = step_size / (llc_tau + step_size)
prev_filtered_forces = np.array([0.0, 0.0, 0.7 * 9.81 / 4] * n_carriers)

Fz = 0.7 * 9.81 / 4
desired_forces = [0.0, 0.0, Fz] * n_carriers  # only used if bypass_controller=1


# ------------------------------------------------------------------ #
#  Build the environment and read the FMU parameters                  #
# ------------------------------------------------------------------ #
env = FMUPlantEnv(fmu_filename="Base_Model.fmu", n_carriers=n_carriers,
                  step_size=step_size, end_time=end_time, load_inertia=0.01)

obs, info = env.reset()

# The controller/optimizer need a few FMU parameters. They are constant, so read
# them once via the env's fmu handle + name->address dict (no new plumbing needed).
Load_Inertia_Matrix = np.array(
    env.fmu.getReal([env.vrs[f"Load_Inertia_Matrix[1,{i}]"] for i in range(1, 10)])
).reshape((3, 3), order="F")

Attachment_Point_Vectors = np.array(
    env.fmu.getReal([env.vrs[f"Attachment_Point_Vectors[1,{i}]"] for i in range(1, 13)])
).reshape((n_carriers, 3))

Load_Mass = env.fmu.getReal([env.vrs["Load_Mass"]])[0]
Cable_Resting_Length = env.fmu.getReal([env.vrs["Cable_Resting_Length"]])[0]

casadi_solver = init_optimizer(Cable_Resting_Length, n_carriers, w_pos, w_vel, phases)


# ------------------------------------------------------------------ #
#  History buffers                                                    #
# ------------------------------------------------------------------ #
time_history = []
drone_vel_norm_history = [[] for _ in range(n_carriers)]
drone_pos_history = [[] for _ in range(n_carriers)]
load_pos_history = []
xi_history, A_history = [], []


# ------------------------------------------------------------------ #
#  Control loop                                                       #
# ------------------------------------------------------------------ #
t = 0.0
start = time_module.time()

while t < end_time - 1e-9:
    # A. Unpack the current observation into named states.
    (curr_pos, curr_orientation_matrix, curr_linVel,
     curr_angVel, drone_pos, drone_vel) = unpack_obs(obs, n_carriers)

    # Record histories.
    time_history.append(t)
    load_pos_history.append(curr_pos.copy())
    for i in range(n_carriers):
        drone_vel_norm_history[i].append(np.linalg.norm(drone_vel[i]))
        drone_pos_history[i].append(drone_pos[i].copy())

    # B. Desired wrench from the PID pose controller.
    ep, eR, ev, ew = error_calculation(curr_pos, curr_linVel,
                                       curr_orientation_matrix, curr_angVel, t)
    w_d = wrench_controller(ep, eR, ev, ew, curr_angVel, Load_Inertia_Matrix,
                            Load_Mass, Attachment_Point_Vectors, step_size,
                            n_carriers, bypass_controller, desired_forces)

    # C. Optimizer -> internal-force sinusoid coefficients.
    lambda_star, f_dot = optimizer(casadi_solver, t, curr_orientation_matrix,
                                   curr_linVel, curr_angVel, w_d,
                                   Attachment_Point_Vectors, epsilon, step_size,
                                   n_carriers, phases, bypass_optimizer)
    xi_history.append(optimizer.prev_x[0])
    A_history.append(optimizer.prev_x[1])

    # D. Distribute the wrench + internal forces into 12 cable forces.
    desired_forces, _ = cable_force_calculation(curr_orientation_matrix,
                                                Attachment_Point_Vectors, w_d,
                                                lambda_star, n_carriers)

    # E. Low-level controller: first-order filter + finite-difference derivative.
    filtered_forces = llc_alpha * desired_forces + (1 - llc_alpha) * prev_filtered_forces
    desired_force_derivatives = (filtered_forces - prev_filtered_forces) / step_size
    prev_filtered_forces = filtered_forces.copy()

    # F. Assemble the 24-D env action [forces(12), derivatives(12)] and step.
    action = np.concatenate([filtered_forces, desired_force_derivatives])
    obs, reward, terminated, truncated, info = env.step(action)

    t += step_size

elapsed = time_module.time() - start
env.close()
print(f"Sim wall-clock time: {elapsed:.5f} s")


# ------------------------------------------------------------------ #
#  Plots (same views as main.py, for visual comparison)              #
# ------------------------------------------------------------------ #
time_history = np.array(time_history)
load_pos_history = np.array(load_pos_history)
drone_pos_history = [np.array(h) for h in drone_pos_history]

# 1. Drone velocity norms.
plt.figure()
for i in range(n_carriers):
    plt.plot(time_history, drone_vel_norm_history[i], label=f"Drone {i+1}")
plt.axhline(epsilon, ls="--", c="gray", label="epsilon")
plt.xlabel("Time (s)"); plt.ylabel("Velocity norm (m/s)")
plt.title("Drone velocity norms over time"); plt.legend(); plt.grid(True)

# 2. XY trajectories.
plt.figure(figsize=(8, 6))
for i in range(n_carriers):
    plt.plot(drone_pos_history[i][:, 0], drone_pos_history[i][:, 1], label=f"Drone {i+1}")
plt.plot(load_pos_history[:, 0], load_pos_history[:, 1], "k--", lw=2, label="Load")
plt.xlabel("X (m)"); plt.ylabel("Y (m)")
plt.title("Drone XY trajectories"); plt.legend(); plt.grid(True); plt.axis("equal")

# 3. Optimizer parameters.
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(time_history, xi_history); ax1.set_ylabel("xi (rad/s)"); ax1.grid(True)
ax2.plot(time_history, A_history); ax2.set_ylabel("A"); ax2.set_xlabel("Time (s)"); ax2.grid(True)
fig.suptitle("Optimizer parameters over time")

plt.show()
