---
name: f2-speed-binding-axis
description: F2 scale-generalization is bound by peak SPEED, not displacement or acceleration. User-tested. Consequence for the mix curriculum: add breadth via new DIRECTIONS at fixed speed; treat speed as its own deliberate curriculum axis.
metadata:
  node_type: memory
  type: project
---

**SCALE-GENERALIZATION IS BOUND BY PEAK SPEED, not displacement or acceleration (user-tested, 2026-09-13).** For a quintic rest-to-rest move of displacement D over duration T: **v_peak = 1.875·D/T**, **a_peak = 5.77·D/T²**. User's two data points already ISOLATE speed as the driver:
- **Case A** (D×2, T×2 -> v_peak ×1.0, a_peak ×0.5): only MILD degradation. This is the control — displacement doubled AND peak accel halved, but speed held -> barely moved. Rules out BOTH displacement-magnitude and acceleration as the cause.
- **Case B** (D×2, T×1.5 -> v_peak ×1.33, a_peak ×0.89): BREAKS more. The only thing that rose vs Case A and past baseline is peak SPEED. So the A->B breakage = the speed increase.
- Baseline regime: MIX_SCALE=5m, MIX_RAMP=16s -> v_peak ≈ 0.59 m/s.

**WHY speed is binding (fixed-wing + cable-suspended load, physical):** (1) AIRSPEED ENVELOPE — carriers have a stall floor (`vmin` metric); formation translation speed shifts each carrier's airspeed/control authority off the trained operating point. (2) LOAD EQUILIBRIUM GEOMETRY — the suspended load is a pendulum that tilts back at speed; the steady cable/lag angle is speed-dependent (P14/P15 "static equilibria of a cable-suspended load with non-stop flying carriers"). A residual trained at one cruise speed learns one equilibrium tilt; higher speed moves the equilibrium and the residual is mis-calibrated. (3) OBS-NORM — obs_mean/std estimated in the training speed band; higher-speed states go off-distribution.

**CONSEQUENCES for the mix curriculum ([[f2-mix-curriculum]]):**
- **Add breadth via new DIRECTIONS at the SAME 5m/16s (v_peak fixed)** = safe breadth, stays in the learned speed band, and it's the symmetry-breaking kind that straightened the field ([[f2-rotation-limited]]). This is the 4th-task recommendation.
- **Displacement breadth (if wanted) = scale D and T TOGETHER (Case A, constant speed)** — only mild degradation. Safe knob for "works at different distances."
- **SPEED is the real generalization frontier and its OWN deliberate curriculum**: warm-start up a ladder of increasing v_peak (raise the D/T ratio in small steps), RE-ESTIMATE obs-norm at each rung. Physically meaningful (deploy across an airspeed envelope) -> likely a genuine thesis contribution. Do NOT casually mix speeds into the direction set.
- **STRUCTURAL:** MIX_DIRS currently carries only direction; MIX_SCALE/MIX_RAMP are GLOBAL. A speed/scale study needs the tuple extended to `(label, dir, scale, ramp)` so tasks can differ in speed. Minor refactor, do it when starting the speed ladder.
- **CRITB COUPLING** ([[f2-mix-curriculum]] result block): adding tasks raises the critical batch (3-traj critB ~1M vs 2-traj ~300-700k); at 3 tasks 1.4M is only ~1.4x supercritical. A 4th task needs STEPS_PER_ITER bumped to ~1.87M (hold ~467k/task) or the clean descent degrades.
