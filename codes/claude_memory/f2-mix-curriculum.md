---
name: f2-mix-curriculum
description: F2 RUNG-3 MIX curriculum (multi-direction joint training) + the critic 76-D desired-load-state fix + the not-conflict-just-slow diagnosis. Current active work as of 2026-09-09.
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-09-09T19:44:51.044Z
---

**RUNG 3 = MIX: train JOINTLY over movement DIRECTIONS (the generalization vehicle).** `MIX_DIRS` in mappo.py, quintic per direction (`make_quintic_pose`, scale `MIX_SCALE`=5m, ramp 16s, horizon `OVERFIT_END`=HOLD+16+1~18s). Builds on [[f2-distillation-capacity-verdict]] (map is REPRESENTABLE) + [[f2-axis-generalization]].

**CRITIC INPUT GREW 58 -> 76 (2026-09-09).** `critic_input()` now appends the 18-D DESIRED LOAD STATE `[p_d, R_d(9), v_d, ω_d]` (via `get_reference_trajectory(env.t, env.traj)`), so state = true(42) + delays(4) + carrier-tgt(12) + desired-load(18). WHY: carrier targets ALONE do NOT determine the load reference — cable-suspended load, so carrier->load is forward-kinematics with a SWING DOF (underdetermined); the critic was inferring p_d from the formation only via the expert's placement convention. Now it gets p_d directly (paired with true load in state) = clean reference side of the load-tracking penalty. Training-only/privileged, actor obs UNCHANGED at 98. Warm-start REINITS the critic (state_dim changed) -> brief EV dip then recovers. (On big multi-traj mixes this matters more; on the easy 2-traj it barely moved EV.)

**THE 5-MIX "not learning / high variance" DIAGNOSIS (decisive chain):**
- NOT critic-identifiability: **EV stable 0.92-0.98** (advantages CLEAN, low-noise). Ruled out.
- NOT the dlam CLIP dead-zone: **satL/satW sub-cap** (below 1). Ruled out (that dead-zone = [[f2-shear-precursor-rewards]]/overshoot term; would zero the dlam gradient if saturated).
- SO the bottleneck is the POLICY UPDATE side, not the critic. Two suspects: (A) multi-task GRADIENT CONFLICT, (B) just SLOW + high stochastic variance.
- **2-TRAJ CONFLICT PROBE** ({+x+y+z, +x+y-z} — same +x+y, only z-sign flips): they **CO-LEARN (not anti-phase)** -> NOT gradient conflict. So the 5-mix failure is BREADTH/VARIANCE + genuine off-axis slowness (RUNG-1 single +y took 150+80 iters), COMPOUNDED by batch-splitting (each traj gets ~half the per-iter gradient).
- FIX = CURRICULUM: solve ONE direction on the FULL batch first (no splitting), save, then re-add the 2nd WARM-STARTED (they co-learn -> 2nd comes up fast). Currently doing +x+y+z solo -> then add +x+y-z. Expand to 5 only once 2 co-learn cleanly (add one dir at a time = rehearsal, avoids the sequential-forgetting seen in [[f2-overfit-adapt-experiment]]).

**SPEED LEVERS (clean advantages -> trust the direction, take bigger steps):** LR_ACTOR is the lever (tried 3e-4->6e-4). BUT 6e-4 accelerated ENTROPY COLLAPSE (4.1->2.5 fast). ENT_COEF small (0..2e-3). Note entropy dip-then-RISE-to-~5 (with ENT_COEF~0) = policy still finding better actions in the tails (mean mis-placed, not stuck) — only bad if it RUNS AWAY or DET_R stalls while it inflates. DEPLOYED policy is the MEAN (DET_R), so entropy=exploration-std matters only via collection quality/runaway. If premature collapse: log_std FLOOR (clamp min after opt step) decouples mean-speed from std-collapse; if runaway: log_std CEILING. Bigger batch does NOT help here (variance already low, EV high) — wrong lever.

**TOOLING added this session (in git):** scale_test.py — loopMSE/loadMSE columns (m²) + on-blowup prints the EPISODE-MEAN-until-blowup stats (loop/load/MSE) instead of nothing; determinism confirmed (seeded EVAL_SEED, mean policy) — irreproducible old figures were config drift (EVAL_DELAYS [1,2,2,1]->[2,2,2,2] on Aug 28, imported from mappo). deploy_compare.py — QUINTIC MOVE mode (dial dir/mag/ramp like scale_test) + load MSE / carrier-vs-optimizer MSE / lambda-vs-opt MSE table + tracking-error & carrier-deviation time plots + move_descr titles (dir·mag not label). mappo.py — coll/upd timing split. Parallel-collection infra: see [[f2-hpc-migration]].
