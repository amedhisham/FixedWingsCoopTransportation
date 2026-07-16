"""
classical_agent.py
===================
A stateful, instance-based wrapper around the EXISTING classical controller +
optimizer, so we can run several *independent* copies at once (one per drone).

Why this exists
---------------
controller.py and optimizer.py keep their state as *function attributes*:
    wrench_controller.intg_ep, optimizer.prev_x, optimizer.prev_lam, ...
Those are process-wide singletons - fine for the single centralized controller
in main.py, but impossible to have four of. This class moves that same state
onto `self`, while reusing the original PURE functions untouched:
    error_calculation, calculate_grasp_and_nullspace, cable_force_calculation,
    init_optimizer.

One ClassicalAgent == one full "centralized" controller+optimizer that runs on
whatever (possibly noisy) state it is handed, and returns the FULL 3n force
vector. In the decentralized stitch, the wrapper keeps only this agent's own
3-D slice. (This is a temporary shim; we'll clean up controller.py/optimizer.py
into proper classes later.)

Run as a single instance on clean state, it reproduces main.py exactly.
"""

import numpy as np

from controller import error_calculation
from optimizer import (
    calculate_grasp_and_nullspace,
    cable_force_calculation,
    init_optimizer,
)


class ClassicalAgent:
    def __init__(
        self,
        n_carriers,
        step_size,
        phases,
        epsilon,
        cable_resting_length,
        load_mass,
        load_inertia,          # 3x3 matrix
        attachment_points,     # (n_carriers, 3)
        solver=None,           # share one CasADi solver across agents if given
        xi0=2.0, A0=1.2,       # initial (frequency, amplitude) of the sinusoids
        delta_xi=0.05, delta_A=0.05,   # per-step monotone increase caps
        w_pos=0.0, w_vel=0.0,  # optimizer cost weights (only used if building solver)
    ):
        self.n = n_carriers
        self.dt = step_size
        self.phases = np.asarray(phases, dtype=float)
        self.epsilon = epsilon
        self.L0 = cable_resting_length
        self.mass = load_mass
        self.inertia = np.asarray(load_inertia, dtype=float)
        self.Bb = np.asarray(attachment_points, dtype=float)   # (n, 3)

        self.xi0, self.A0 = xi0, A0
        self.delta_xi, self.delta_A = delta_xi, delta_A

        # PID gains, copied verbatim from controller.wrench_controller.
        self.Kp = 5.0 * np.eye(3)
        self.Kv = 2.0 * np.eye(3)
        self.Ki = 0.9 * np.eye(3)
        self.KR = 0.5 * np.eye(3)
        self.Kw = 0.06 * np.eye(3)
        self.KiR = 0.1 * np.eye(3)
        self.g = 9.81
        self.e3 = np.array([0.0, 0.0, 1.0])

        # The CasADi/IPOPT solver is stateless between calls (all history is passed
        # in as parameters), so several agents may safely share one instance. If
        # none is given, build a private one.
        self.solver = solver if solver is not None else init_optimizer(
            cable_resting_length, n_carriers, w_pos, w_vel, self.phases
        )

        self.reset()

    # ------------------------------------------------------------------ #
    #  Episode state                                                     #
    # ------------------------------------------------------------------ #
    def reset(self):
        """Zero the integrators and re-seed the optimizer history (call per episode)."""
        # Wrench-controller integral terms.
        self.intg_ep = np.zeros(3)
        self.intg_eR = np.zeros(3)

        # Optimizer sinusoid state, seeded exactly like optimizer.py's first call.
        self.prev_x = np.array([self.xi0, self.A0])
        self.prev_lam = self.A0 * np.cos(self.xi0 * 0.0 + self.phases)
        self.prev_lam_dot = -self.A0 * self.xi0 * np.sin(self.xi0 * 0.0 + self.phases)

        # Derivative-history terms; seeded lazily on the first optimize() call so
        # the first finite difference is zero (matches optimizer.py behaviour).
        self.prev_w_d = None
        self.prev_G_pinv = None
        self.prev_N = None

    # ------------------------------------------------------------------ #
    #  Outer-loop wrench controller (eqs. 14-15)                          #
    # ------------------------------------------------------------------ #
    def wrench_control(self, ep, eR, ev, ew, ang_vel):
        self.intg_ep += ep * self.dt
        self.intg_eR += eR * self.dt

        f_L_d = (self.mass * self.g * self.e3
                 - self.Kp @ ep - self.Kv @ ev - self.Ki @ self.intg_ep)

        gyroscopic = np.cross(ang_vel, self.inertia @ ang_vel)
        tau_L_d = (gyroscopic
                   - self.KR @ eR - self.Kw @ ew - self.KiR @ self.intg_eR)

        return np.concatenate((f_L_d, tau_L_d))

    # ------------------------------------------------------------------ #
    #  Optimization layer (eqs. 22-25)                                   #
    # ------------------------------------------------------------------ #
    def optimize(self, t, R, lin_vel, ang_vel, w_d, bypass=False):
        _, G_pinv, N = calculate_grasp_and_nullspace(R, self.Bb, self.n)

        # Lazy seed of the derivative history: first step -> zero derivatives.
        if self.prev_G_pinv is None:
            self.prev_w_d = w_d
            self.prev_G_pinv = G_pinv
            self.prev_N = N

        # Finite-difference derivatives of the time-varying terms.
        G_pinv_dot = (G_pinv - self.prev_G_pinv) / self.dt
        N_dot = (N - self.prev_N) / self.dt
        w_d_dot = (w_d - self.prev_w_d) / self.dt

        # External contribution e(t) (eq. 20).
        e_total = (G_pinv_dot @ w_d) + (G_pinv @ w_d_dot)

        # Base carrier velocities v_Li (eq. 22 base term).
        v_L_stack = []
        for i in range(self.n):
            v_Li = lin_vel + R @ np.cross(ang_vel, self.Bb[i, :])
            v_L_stack.extend(v_Li)
        v_L_stack = np.array(v_L_stack)

        # Pack every parameter into the flat vector the solver expects.
        p_val = np.concatenate([
            [t], self.prev_x, self.prev_lam, self.prev_lam_dot,
            w_d, G_pinv.flatten("F"), N.flatten("F"), N_dot.flatten("F"),
            e_total, v_L_stack,
        ])

        lbg = [self.epsilon ** 2] * self.n

        if bypass:
            opt_x = np.array([2.0, 1.2])
        else:
            # xi and A may only increase (monotone), capped per step (see optimizer.py).
            lbx = [self.prev_x[0], self.prev_x[1]]
            ubx = [self.prev_x[0] + self.delta_xi, self.prev_x[1] + self.delta_A]

            res = self.solver(x0=self.prev_x, p=p_val,
                              lbx=lbx, ubx=ubx,
                              lbg=lbg, ubg=[np.inf] * self.n)

            if self.solver.stats()["return_status"] == "Solve_Succeeded":
                opt_x = np.array(res["x"]).flatten()
            else:
                opt_x = self.prev_x.copy()

        opt_xi, opt_A = opt_x[0], opt_x[1]

        # Analytical sinusoids and their derivatives (eq. 23).
        lambda_star = opt_A * np.cos(opt_xi * t + self.phases)
        lambda_star_dot = -opt_A * opt_xi * np.sin(opt_xi * t + self.phases)

        # Cable-force derivative (eq. 19).
        f_dot = e_total + (N_dot @ lambda_star) + (N @ lambda_star_dot)

        # Roll history forward.
        self.prev_x = opt_x
        self.prev_lam = lambda_star
        self.prev_lam_dot = lambda_star_dot
        self.prev_w_d = w_d
        self.prev_G_pinv = G_pinv
        self.prev_N = N

        return lambda_star, f_dot

    # ------------------------------------------------------------------ #
    #  One full control step: state  ->  full 3n cable-force vector       #
    # ------------------------------------------------------------------ #
    def compute_forces(self, load_pos, lin_vel, R, ang_vel, t, bypass_opt=False):
        """Run error -> wrench -> optimizer -> distribution on the given state.

        Returns the FULL (3n,) desired cable-force vector and its (3n,) analytic
        derivative. The caller decides which slice to keep and whether to apply
        the low-level actuator filter.
        """
        ep, eR, ev, ew = error_calculation(load_pos, lin_vel, R, ang_vel, t)
        w_d = self.wrench_control(ep, eR, ev, ew, ang_vel)
        lambda_star, f_dot = self.optimize(t, R, lin_vel, ang_vel, w_d, bypass=bypass_opt)
        forces, _ = cable_force_calculation(R, self.Bb, w_d, lambda_star, self.n)
        return forces, f_dot
