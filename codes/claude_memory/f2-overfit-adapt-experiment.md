---
name: f2-overfit-adapt-experiment
description: "F2 ACTIVE experiment (2026-08-29): warm-start OVERFIT adaptation on [+y quintic, +x+y custom] to test if +y is solvable when NOT a dodgeable minority — the decisive test of the exploration/dodge diagnosis"
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-09-01T11:25:47.130Z
---

**THE ACTIVE EXPERIMENT (2026-08-29): warm-start ADAPTATION / OVERFIT.** Decisive test of the [[f2-axis-generalization]] diagnosis (the +y wall is EXPLORATION/DODGE, not capacity): **can the net be FORCED to solve +y when +y is NOT a dodgeable minority?**

**SETUP (mappo.py `OVERFIT` mode, wired 2026-08-29):**
- `OVERFIT=True` -> ignores the 55-traj distribution; uses `overfit_set()` = 2 fixed trajs: **+y quintic** (1 m, ramp 12, hold 2 — the scale_test case, computed once via expert_path since not in lib) + **+x+y custom** (`training_pairs()[2]`, precomputed in expert_lib). Eval on the SAME two (`n_anchor=len(pairs)` -> both sampled every iter). `yqui`/`xy_c` in the DET spread.
- **WARM-START from `residual_mappo.pt`** (the current net) — the point is to FORCE the +x-specialist to adapt, and separately measure how much +x it FORGETS. (Set `WARMSTART=None` for the cold variant.)
- **Saves to `residual_mappo_overfit.pt` / `_overfit_last.pt`** — PROTECTS the main policy AND the warm-start source (don't clobber them).
- FULL current reward ON (overspeed + overshoot both kept — see below). 256-wide, delay [2,2,2,2].

**WHY +x+y (not custom +y):** custom +y — the BASE ITSELF blows up (no clean target to fall back to). +x+y base is STABLE (loop 1.141) so there's a good reference, AND a COMBINATION direction forces sharing / teaches generalization (not pure-axis memorization).

**REWARD DECISION (resolved): KEEP OVERSHOOT ON.** Initially I wanted overshoot OFF (feared it starves +y of dlam authority) — WRONG. The overshoot hinge is ZERO for sat_lam<=1 (FULL cap authority is free); it only bites sat>1, which is CLIPPED away anyway (applied dlam = cap whether raw is 1.0 or 1.5). Its equilibrium is sat~=1 (max applied authority, no waste) -> does NOT starve authority. It discourages the whack-a-mole deep saturation that PRECEDES the shear, and aligns with the oracle (dlam*~=0 -> the correct +y solution is LOW dlam). So it nudges toward the stable basin. overspeed is the DIRECT blowup deterrent, overshoot the INDIRECT — both pull the same way. See [[f2-shear-precursor-rewards]].

**DECISION TREE (what the run tells us):**
- **yqui/xy_c loops FALL + blowups DROP** -> +y IS solvable when forced -> **DODGE confirmed** -> the joint fix is CURRICULUM / up-sample +y (make +y non-dodgeable), NO depth, NO distillation.
- **Can't adapt even with all focus** -> deeper than dodge — reachability/basin or fundamental (reward/plant/obs) -> heavier interventions (specialist-distill, or re-examine the setup).
- Converges FAST (2 trajs, 56k steps/iter, EPOCHS=10) -> verdict in ~10-20 iters, no need for all 150.

**AFTER it adapts:** point `scale_test.py CKPT=residual_mappo_overfit.pt` and check **+x** (+ the sweep) to measure FORGETTING — the whole "how much does forcing +y cost +x" question.

**READY-TO-RUN status (double-checked 2026-08-29):** no crash (consist-off path safe: `_wd_clean`/`_lam_clean`/`_oracle_mode` init to None + guarded); warm-start matches (residual_mappo.pt is 256, HIDDEN=(256,256), obs 98); saves protected; overfit_set builds (both dpos (4,3500,3)). Just set `OVERFIT=True` and run.

=================================================================================
**OUTCOMES (2026-08-31..09-01) — the OVERFIT experiment became a FEASIBILITY CURRICULUM and gave clean answers:**
- **A crash fix first:** eval_policy KeyError 'sat_lam' on blowup steps (blowup info omits sat_lam/sat_w) — added `break` on blowup + guarded empty post-loop stats. Only bit OVERFIT because its eval trajs blow at warm-start.
- **The incentive REDESIGN (key):** dense overspeed penalty + fixed -100 terminal makes surviving-then-blowing WORSE than blowing early (survival just accrues more penalty) -> early-death gradient. FIX = make a no-blowup outcome REACHABLE. Plus un-froze the dlam head: it was parked in the CLIP dead-zone (raw dlam > cap -> applied clipped -> ZERO task gradient on magnitude), fixed by cap_lam 0.5->0.65 (current op point becomes sub-cap -> gradient restored) + overshoot_lam_w 1->10 (strong enough to un-saturate vs overspeed). See [[f2-shear-precursor-rewards]].
- **RUNG 1 (gentle +y quintic, ramp 16, SHORT 19 s horizon):** SOLVED — loop 5.0 sentinel -> ~0.24, satL 0.37 (head un-frozen), 0 blowups. Took 150+80 iters (a LOT for one feasible task -> off-x is genuinely hard even when reachable). +x forgot only ~0.01 (single-task adaptation holds +x).
- **RUNG 2 (+x+y custom, FULL 35 s):** SOLVED (leans on the dw/range head satW~0.69, dlam quiet satL~0.32 = oracle-aligned) BUT **COOKED the +x and +y quintics** = catastrophic forgetting from SEQUENTIAL single-task fine-tuning (no rehearsal — old tasks aren't in the loss).
- **JOINT {+y quintic, +x+y custom}:** did NOT learn well (blowups late -> the incentive trap again). This looked like a capacity/conflict wall...
- **...but DISTILLATION settled it (2026-09-01): NOT capacity, it's PPO.** One net fits both expert maps R^2~0.99. Full verdict + the seed-from-BC fix in [[f2-distillation-capacity-verdict]]. LESSON: sequential single-task rungs FORGET (need rehearsal/joint); and the residual can REPRESENT multi-direction — the wall is RL exploration on a blowup-prone landscape.
- Widened `residual_mappo_r4base` old-arch net (obs 44/hidden 128) up to current arch (obs 98/hidden 256, function-preserving) -> `residual_mappo_r4base_wide.pt` (scratchpad script widen_r4base.py chains widen_checkpoint's input-widen + widen_hidden's hidden-widen; actor function-exact, critic reinits on mappo warm-start since its state grew).