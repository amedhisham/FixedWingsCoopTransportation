---
name: project-overview
description: "Fixed-wing cooperative cable-suspended load transport — architecture, key equations, and file roles"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4cd4cd58-4f95-4b55-b733-ca69719ced16
---

Implements cooperative transport of a cable-suspended load by n fixed-wing carriers (drones that cannot stop — stall constraint).

**Why:** Fixed-wing UAVs must maintain minimum forward speed ε (non-stopping constraint), unlike multirotor. This requires an optimizer that picks internal cable forces keeping every carrier moving.

**Key references:**
- `references/p2025ut_non_stop_carriers_moving_load.pdf` — main paper (equations referenced inline in code)
- `references/Girardello_Sofia.pdf` — thesis providing deeper derivations
- Related: P14 (static equilibria), P15 (coordinated trajectories)

**Architecture:**
- `main.py`: 100Hz co-simulation loop with Simulink FMU (Base_Model_three_drones.fmu). Reads load/carrier states, runs controller → optimizer → cable_force_calculation, injects results back.
- `controller.py`: Geometric PID+I wrench controller. Computes desired wrench w_d = [f_Ld, τ_Ld]. Reference trajectory: hover 5s → move 0.5 m/s for 10s → stop.
- `optimizer.py`: Core contribution. Grasp matrix G (Eq 8), nullspace N (structured, Hamiltonian cycle), CasADi NLP for (ξ, A) optimization.

**Cable force decomposition:**
  f = G†(R,t) w_d + N(R,t) λ
  G is 6×3n, rank 6 → nullspace dim = 3n-6

**Parametric internal forces (Eq 23):**
  λ_j(t) = A cos(ξt + φ_j)   — common ξ, A; phase shifts φ_j fixed per carrier

**Non-stopping constraint (Eq 24):**
  ||v_Ri||² ≥ ε²  for all carriers i
  v_Ri = v_Li + (L/T_i) Π_i (ė_ext + ġ_internal)
  where v_Li = v_L + ω_L × (R_L b_i)   ← the selected lines 207-208 of optimizer.py

**Optimization (Eq 20):**
  min over (ξ, A): smoothness cost + ||λ - λ_prev||² + ||λ_dot - λ_dot_prev||²
  s.t. ||v_Ri||² ≥ ε²

Current state (branch: three-drones): 3-carrier version. bypass_optimizer=1 and bypass_controller=1 flags exist for debugging with static controller.

**How to apply:** When suggesting changes, keep equation numbers as comments since they tie directly to the paper. The structured nullspace uses a Hamiltonian cycle (not scipy null_space) to ensure it stays in the correct subspace analytically.
