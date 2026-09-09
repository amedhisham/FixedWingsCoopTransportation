---
name: f1-antimemorization-plan
description: "F1 next-steps: buzz RESOLVED (in-loop EMA), optimizer ratchet RESOLVED (deadband), and the trajectory/timing plan to stop the net memorizing the clock"
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-08-14T17:31:43.551Z
---

Captured 2026-08-14, right before a compaction. Continues [[f1-milestone]]. Two threads got
resolved this session, and a concrete implementation plan is queued (NOT yet done).

## RESOLVED 1 — the F1 closed-loop BUZZ (the long-open issue)
Root cause chain (all confirmed): the reconstructed pR_dot feeds back the net's OWN lambda, and
`lambda_dot = dlam/dt` (÷DT ≈ ×100) amplifies any step jitter into a self-excited limit cycle.
It's a **marginal-stability** loop: loop gain `(1-beta)*G0` with G0 ≈ 1.0-1.23; buzz onsets sharply
when gain crosses 1 (that's why 10% expert flips it, and why DAgger to beta=0 is fragile).
**Imitation CANNOT fix it** — MSE is blind to gain/stability, and more low-beta data makes it WORSE
(the buzz corrupts the pR_dot input → the imitation target becomes ill-posed → MSE rises WITH buzz).
The optimizer stays stable in the SAME reconstruction loop only because its **warm-started solve
enforces continuity** (lambda can't jump); the memoryless MLP lost that.
**FIX = in-loop output EMA (soft warm-start continuity), Design 1.** In `dagger_prdot.rollout` the
APPLIED lambda = `a*mixed + (1-a)*prev`, `a=DT/(tau+DT)`, and the smoothed lambda drives the plant
AND feeds back (recon/prev_lam) — so the net trains against the SAME filtered feedback it sees at
deploy (NOT a post-hoc bolt-on, which detuned the loiter into a "wacky" wander). Shared const
`LAM_LP_TAU = 0.01` in `collect_prdot_data.py` (imported by dagger rollout + `deploy_prdot`).
RESULT: retrained BC→DAgger(0.9→0), beta=0 buzz 0.0195 (floor), MSE 0.0006, track 0.0166. STABLE.

## RESOLVED 2 — optimizer A/xi drift ("sus #1")
The optimizer's A (amplitude) and xi (frequency) crept up all episode even though the constraint was
NEVER near binding (min analytic ||v_Ri|| = 1.065 vs eps=0.25). NOT physics, NOT the v_Ri mismatch.
Cause: cost is pure smoothness `(xi-prev)²+(A-prev)²` with `w_pos=w_vel=0`, so the true optimum IS
"don't move"; but IPOPT returns prev ± tolerance, and the **monotone lower bound (A/xi may only
increase) RECTIFIES that solver slop into a one-way ratchet** (~5e-5/step → 0.13 over 25s). The user
confirmed allowing decrease is NOT an option (degenerate solutions).
**FIX = DEADBAND in `classical_agent.optimize`**: compute analytic ||v_Ri|| at prev_x (Eq 22, matches
`optimizer.py` constraint) BEFORE solving; if `min > solve_below` (=0.3 = eps+margin) → HOLD prev_x,
skip the CasADi solve. Only solve when genuinely near-binding. New: `self.solve_below=0.3` ctor arg,
`self.last_vRi` (=√res["g"], the constraint velocity), `self.solved` flag. RESULT: A/xi dead-flat on
gentle quintics, solver ~never engages → **big collection speedup**, and it gives real CONSTANT-A/xi
trajectories (needed for the anti-memorization below).

## "sus #2" — the clock is a POSITIONAL ENCODING (memorization risk) — but it REPEATS at 25.13s
The clock (`collect_il_data.clock_features`, 7 freqs ω=arange(1.5,3.0,0.25), 14-D sin/cos) uniquely
fingerprints absolute time ONLY within one period. CORRECTION (user, 2026-08-14): the freqs are ALL
integer multiples of 0.25 rad/s, so gcd=0.25 → the joint vector REPEATS EXACTLY every 2π/0.25 = 8π
≈ 25.13 s. (My earlier "freqs don't realign / unique over the whole episode" was WRONG.) So with
T_END=35 the tail [25.13, 35] (~9.9s) is a VERBATIM replay of [0, 9.9]: the net CANNOT use the clock
to disambiguate those states → must read pR_dot/prev_lam → the last ~10s is a FORCED no-memorization
zone (for free, just by T_END > clock period). It only bites where the tail has MOTION (else the
aliased-early λ ≈ tail λ, benign) — hence ensuring movement to ~t=30 matters. Division of labor:
randomized quintic timing guards [0,25.13] (where the clock IS still unique/memorizable); the clock
period wall guards [25.13,35]. So the net CAN still memorize a time-indexed A/xi schedule on [0,25],
User's clock design is deliberate + good: many freqs so the net can switch xi and time itself, but
ONLY conditioned on pR_dot nearing eps. It works ONLY if the data has a MIX of constant-A/xi and
adaptive-A/xi trajectories (so pure-time memorization can't fit both). The deadband now provides that
mix; the plan below strengthens it.

## IMPLEMENTED 2026-08-14 (the plan below is now DONE, pending the retrain run)
- `controller.make_linear_move(vel, hold, move_dur, base_pos, base_R)`: piecewise constant-velocity
  factory (generalizes the default straight-line to any direction/timing; velocity STEP at both ends
  = the harsh transient that engages the solver). R fixed.
- `trajectories.py`: `POS_RANGE 1.5->3.0`; `RAMP` (fixed 10) -> `RAMP_MIN,RAMP_MAX=15,25` randomized
  per-quintic (move-end ~U[20,30]s); `custom_set()` returns 5 solver-engaging customs.
- `collect_il_data.T_END 25->35` (single source; F2 scripts inherit it — revisit F2 time-indexed reward).
- `collect_prdot_data.collect()` and `dagger_prdot.main()` both prepend `custom_set()` to every batch.
- `deploy_compare.py`: `USE_CUSTOM` (0..4) to survey a custom traj.
- SHOWCASE infra (test-only demo trajs, NEVER collected; for thesis plots + length-generalization
  proof): `trajectories.showcase_set(kind, M)` -> [(label, traj, t_end)]. `kind="short"` = 25s
  (original +x line move[5,15] + M short quintics ramp=10, move-end t=15 OUTSIDE training's U[20,30]);
  `kind="long"` = 35s training-like (line-at-35 = the default anchor + M quintics ramp=23). Also
  `showcase_line(t_end)`, `showcase_quintics(M,t_end,ramp,seed=SHOWCASE_SEED=424242)` (disjoint seed;
  same seed as heldout would NOT be disjoint — uses its own). ENABLER: `run_episode`(deploy_prdot) +
  `run_episode_opt`(deploy_compare) both take `t_end` (defaults T_END) so any horizon runs; env built
  at t_end. `deploy_prdot.run_showcase(kind)` loops the set (net only, all plots); `deploy_compare`
  SHOWCASE + SHOWCASE_IDX picks ONE for the net-vs-opt overlay. Verified short_line@25s: 2500 steps,
  track 0.081, solve 2.8%.
- **KEY PHYSICS FINDING (verified headless, solve% over a 35s episode, deadband solve_below=0.3):**
  the internal-force coordination is a HORIZONTAL-plane phenomenon. ANY vertical component keeps the
  4-drone formation symmetric and lifts every ||v_Ri|| off the eps-floor -> `+z` 0% (min vRi 1.06),
  `+x+z` 0% (0.58), `+x+y+z` ~0% (0.27). Only in-plane moves engage: `+x` 5.3%, `+y` 5.1%, `+x+y`/
  `-x+y`/`+x-y` 10.2% (default straight-line 2.0%). So the 5 customs are ALL HORIZONTAL (user chose
  "all 5 horizontal"); z-variety is left to the 3-D quintics. Customs = engagement (horizontal),
  quintics = 3-D variety.

## THE PLAN AS ORIGINALLY QUEUED (now implemented above)
1. **Add ~5 "custom" (non-quintic) solver-ENGAGING trajectories.** The default straight-line
   (`controller.get_reference_trajectory(t, traj=None)`: +x at v=1.1, hold 0-5 / move 5-15 / hold)
   is the harsh one that stresses eps. Add ~5 like it but moving in **y, z, and xyz combos** (default
   only moves +x). 5 of ~55 (50 quintic + 5 custom) — enough to NOT drown among the gentle 50 yet
   force the net to learn the adaptive (pR_dot-conditioned) A/xi behavior.
2. **Episode length T_END: 25 → 35 s** (`collect_il_data.py: DT, T_END = 0.01, 25.0`, imported
   everywhere). MUST have MOVEMENT up to ~35s on some trajectories, NOT pure loiter after a fixed
   time (else the net memorizes "loiter starts at t=X"). Custom trajectories: ramp/move until ~t=30.
3. **Quintics: break the fixed timing.** `trajectories.py` currently POS_RANGE=1.5, RAMP=10, HOLD=5
   (move fixed 5→15). Increase POS_RANGE so moves extend past 15s, and **RANDOMIZE the move-end time
   ~U(20,30)s** across trajectories so there's no fixed "move ends at 15" marker to memorize. Keep the
   held-out seed disjoint.
4. **RETRAIN end-to-end**: re-collect (fast now via deadband) → `train_prdot` (BC) → `dagger_prdot`
   (0.9→0, in-loop EMA). Consider re-enabling `USE_CURATION` (currently False from the poisoning A/B).

## Infra state (so post-compaction I know what exists)
- Net: `PRDOT_HIDDEN=(256,256)`, ckpts self-describe via saved "hidden" key.
- Hardness curation in `collect_prdot_data`: `keep_probs`/`curate`/`cap_indices` (two-level: activity
  bins loiter/cruise/transition via ref speed+accel, per-bin TRUE keep floor `P_MIN=0.15`, graduated
  deciles) + `recency_weights` (`RECENCY_GAMMA=0.85`, `age` persisted). `dagger_prdot`: `TRAJ_PER_ITER=30`,
  `MAX_AGG=150k`, `WARM_START`, `USE_CURATION` toggle (A/B'd the poisoning; found it's NOT curation —
  it's imitation-can't-stabilize; currently False). Seeds off `gen0` so staged/chunked runs continue.
- `deploy_compare.py`: net-vs-optimizer on ONE trajectory. Toggles `RUN_NET` (False=optimizer-only,
  fast), `USE_DEFAULT` (default straight-line), `HELD_IDX`. Plots velocity (plant, dashed eps), lambda,
  XY, and optimizer A/xi; prints `solver engaged %`, closest v_Ri approach. `run_episode_opt` reused.
- `diagnose_buzz.py`: extended-hover buzz FFT/growth/tension/pR_dot diagnostics (how the buzz was
  characterized). `LAM_LP_TAU` there is independent scratch.
- Current status: F1 generalizes (held-out track cm-level, buzz at floor). The plan above is to make
  the generalization ROBUST to timing (not clock-memorized) and to exercise the adaptive optimizer.
