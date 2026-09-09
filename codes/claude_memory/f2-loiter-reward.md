---
name: f2-loiter-reward
description: "The loitering/circulation reward design for F2 — replace expert-path tracking with \"orbit your attachment fast enough that stall is geometrically impossible\""
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-08-24T16:28:20.274Z
---

**WHY (diagnosis, 2026-08-24):** F2 residual stall is MID-FLIGHT (not startup) and NOT structural — IDENTICAL turns behave differently (some fine, some stall). So it's STATE/DESYNC-dependent: the drone enters some turns mis-coordinated -> stall there. Reward-tuning is EXHAUSTED as a lever: cranking stall_w (even 400/lin10) steamrolls+thrashes without fixing stall; setting manifold_w=0 (kill path-tracking) STILL stalls -> proves it's NOT reward specification, it's optimization/reachability (warm-start gt2 flat gradient critic_loss~0.007 + the "speed up through the turn" action slackens a cable -> tension collapse -> BLOWUP, so the faster direction is blocked). Actuator-obs idea (feed applied force/f_dot) REJECTED: the LLC is a black box (command->actual delay is exactly what's unknown IRL); the drone sees its physical state, that must suffice. See [[f2-estimation-dead-end]], [[f2-training-log]], [[f2-residual-rl-plan]].

**THE LOITER REWARD IDEA:** stop targeting a POINT on the expert path (which itself dips to ~0.2 m/s at turns -> tracking it FORCES stall). Instead target ORBITING the (moving) attachment fast enough that airspeed can't drop below stall — by GEOMETRY, not by penalty. manifold_w=0 (no expert-path term); the loiter is fully defined by radius+circulation+cable, anchored to the attachment. Warm-start gt2 into it.

**GEOMETRY (per drone i, computed in step() from TRUE state):**
- `c_i = npos + nR @ Bb[i]`  — attachment point in WORLD (load pos + load-rotation applied to body-fixed attach vector Bb[i]). MOVES with the load.
- `r = p_i - c_i`  — drone position relative to its attachment ("spoke"). ||r||≈L0 (cable). Horizontal part = where on the circle.
- `r_xy=r[:2]; rn=||r_xy||` (current horizontal loiter RADIUS); `r_hat=r_xy/rn` (radial, points anchor->drone).
- `t_hat=[-r_hat_y, r_hat_x]`  — r_hat rotated 90° = TANGENTIAL (around the circle). Sign = orbit direction.
- `v_rel = v_i - nvel`  — drone velocity RELATIVE to the load (so we measure orbiting WITHIN the load frame, separate from the whole formation translating). (Rigorous: also subtract omega_load × (nR@Bb[i]); negligible unless load spins fast.)
- `s_tan = v_rel_xy · t_hat`  — orbital (tangential) speed = how fast it goes AROUND.
- `s_rad = v_rel_xy · r_hat`  — radial speed = spiraling in/out (should be ~0 on a circle).

**REWARD TERMS** (added to reward, i.e. penalties; keep stall/overspeed/load guards):
- `radius_w * (rn - RHO)^2`  — stay on the circle of size RHO (two-sided). This + cable = the ANCHOR that replaces manifold (drone can't fly off) -> manifold_w=0 has no degeneracy.
- `circ_w * relu(V_ORBIT - s_tan)^2`  — ONE-SIDED: orbit AT LEAST V_ORBIT. 0 penalty if faster. This is the anti-stall core: forces brisk circling but leaves freedom to speed up through a bad turn (two-sided would punish the extra speed you WANT).
- `radial_w * s_rad^2`  — keep it a circle, not a spiral (also stops gaming the radius by oscillating in/out).

**THE KEY TRICK — why stall dissolves by geometry:** absolute airspeed = ||v_i|| = ||nvel + v_rel||. Worst phase = orbital vel opposes load motion: ||v_i||_min ≈ |s_orbit - v_load|. For NEVER-stall (||v_i||≥eps always): s_orbit ≥ eps + v_load. So set **V_ORBIT ≥ eps + max_load_speed** (with ~1.5x margin). Then circ forces s_tan≥V_ORBIT -> absolute airspeed ≥ eps at EVERY orbit phase and EVERY turn -> stall is geometrically impossible. This is why it beats stall_w: instead of penalizing the symptom on an unreachable/cliff-blocked action, it makes "keep circling briskly" the objective, which HAS a stall-free optimum. Dissolves "identical turns differ" because the anti-stall guarantee is a geometric orbital-speed floor, independent of entry state/desync/phase.

**PARAMS (design constants, computed OFFLINE from the expert loiter = a mission spec, NOT privileged runtime info):** RHO = expert loiter's mean horizontal radius (or a bit larger for margin). V_ORBIT = eps + max_load_speed, margin ~1.5x. Phase separation is FREE (each drone orbits a different c_i via different Bb[i] -> naturally spread; no inter-drone term). load_w couples the 4 circles into jointly holding the load. Does NOT fix the coordination LEAK (separate wall) — only removes the stall symptom, and should be more desync-robust (local "orbit fast" needs no tight inter-drone timing).

**WATCH:** (1) one-sided circ has no upper speed bound -> could spin up absurdly; overspeed+load should cap, else add mild two-sided or lower overspeed thr. (2) orbit direction sign (t_hat) must match the loiter + avoid collisions. (3) world-z orbit plane assumes load ~level; if it tilts, use load-LOCAL horizontal. (4) tune RHO/V_ORBIT: too high = overspeed/hard to hold load, too low = stall not dissolved. IMPLEMENT as new reward terms in residual_marl_env.py, params OFF by default (clean diff); anneal manifold_w DOWN / circ UP if warm-starting so there's never a gap with only penalties.
