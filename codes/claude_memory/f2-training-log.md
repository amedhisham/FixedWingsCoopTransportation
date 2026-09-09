---
name: f2-training-log
description: "Ordered log of F2 two-head residual training regimes + HARD-WON LESSONS (exploration-trap/clip, reward-balance, entropy tuning, swing carried by load term). Current working config + regime 4 (surpass expert) planned."
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-08-02T17:01:01.851Z
---

**F2 two-head residual — ordered training log.** See [[f2-residual-rl-plan]] for the deep rationale.
Each regime is WARM-STARTED from the previous best (resume via WARMSTART=residual_mappo_last.pt).

**FOUNDATION (the architecture that made anything work):** two-head residual action
[delta_lambda(nullspace, 4) + delta_wrench(range, 6)] built as f_base + [G+ dw + N dlam] per drone;
clock in obs (44-D); TIME-INDEXED expert tracking; per-drone reward + per-drone GAE; batch
STEPS_PER_ITER 10k->20k; DETERMINISTIC held-out eval (EVAL_SEED=4242, prints DET_loop/DET_load/
loadmax/swing); blowup guard (truncate if speed>100 or NaN); graceful Ctrl+C -> residual_mappo_last.pt
+ resume. This split BROKE the tracking-vs-load Pareto see-saw -> fixed-scenario DET_loop 0.18, load calm.

**REWARD REGIMES:**
1. expert-track(manifold_w=1) + stall-LIGHT(stall_w=50, floor eps=0.25) + load(load_w=10).
   FIRST ACCEPTABLE results. Fixed-scenario DET_loop 0.18; held-out ~0.28. Then moved to warm-started
   DOMAIN RANDOMIZATION (delays {1,2} per drone + random noise seed, STEPS_PER_ITER 20k).
   - DETOUR (reverted): added SWING term = load-VELOCITY error swing_w=10 (+ cap_w 0.12->0.2) for load
     smoothness. BACKFIRED: cheapest way to cut load velocity is to SLOW the formation -> drones dropped
     below stall. LESSON: a load-velocity penalty fights airspeed; don't naively re-add it. Set swing_w=0.
2. stall cranked 50 -> 300 (swing off) -> deep stalls MOSTLY gone, better.
3. stall_w=400 + LINEAR HINGE (penalty = stall_w*(d^2 + d), d=relu(eps+margin - v); linear term gives
   gradient AT the boundary, quadratic alone ~0 there) + stall_margin=0.05 (cruise floor 0.30, punish
   APPROACHING eps not just below) + stall_grace=20 (skip startup spin-up-from-rest). <-- RUNNING NOW.
   WATCH: team_ep_R climbs+flattens = stalls eliminated (proxy); confirm on demo velocity plot (drones
   stay >0.30). If team_ep_R flat but stalls persist -> NOT incentive: it's authority (raise cap_lam) or
   the REFERENCE (expert itself dips to ~0.2 at sharp turns -> gentle the expert). ~50-100 iters expected.
3b. COORD term ||sum_i f_int||^2 (internal-force leak; ~0 iff drones coordinate). coord_w=50 -> DISASTER:
    team_R -809k, it STEAMROLLED (coord dropped 0.79->0.47 but loop 0.28->0.45, load/swing all WORSE) by
    driving drones OFF the loops (cheapest way to cancel forces = leave them). w=3 gentle retry also weak.
    VERDICT: the FORCE-leak is NOT the swing driver; coord term ABANDONED (coord_w=0). Leak is spike-
    dominated (median 0.03N, max 2.5N); policy adds ~10x base leak but killing it didn't help swing.
3c. JERK smoothness = per-drone ||v_t - 2v_{t-1} + v_{t-2}||^2, STATE-based, startup-graced. First try
    jerk_w=8 UNCLIPPED -> team_R -350k, reward "improved" only via ENTROPY COLLAPSE (exploration died,
    det jerk stayed flat). FIX = CLIP per-step (jerk_cap=0.05): exploration inflates jerk ~30x, clip zeros
    the gradient on the spikes so the ONLY way to cut the penalty is smoothing the MEAN's structural jitter.
    With clip + ENT_COEF fixed, jerk FINALLY drifts down (0.065->0.062). Jerk is NOT the swing lever either
    (swing improves without it) but KEEP IT anyway: smoothness is a goal in itself (real-drone actuators).
4. (STILL PLANNED, RISKY, LAST) tune DOWN manifold_w to a SMALL ANCHOR to SURPASS the expert. Can
   DEGENERATE (all-zero/collapse) -> keep stall+load as guards, do incrementally. Collision-avoidance
   DEFERRED (even classical doesn't guarantee it).

**STALL_W REBALANCE (critical correction):** stall_w=400 was OVERKILL -> it drowned every other term
(jerk/swing became rounding errors) AND the huge penalty pushed the policy to KILL EXPLORATION (ent
4->-0.2). Dropped to stall_w=50: the HINGE+MARGIN already give a firm bite (~8.6 at a 0.15 dip), so 50
enforces the floor WITHOUT steamrolling. The expert dips to ~0.2 at sharp turns (marginal reference) ->
policy approximates -> deeper stall; the floor makes it round the corner (deviate from expert, which is OK).

**ENTROPY (ENT_COEF) tuning:** 0.0 -> COLLAPSE (stall penalty pushes log_std to death, ent-> -0.2,
premature convergence). 0.01 -> RUNAWAY (ent 4.25->5.4). 0.003 = BETTER but STILL climbing slowly (not
stopping) -> NEXT RUN DROP TO 0.002 (or 0.001). Root cause: task gradient is tiny (critic_loss ~0 =>
small advantages) so even a small entropy bonus out-pushes it. Want ent to PLATEAU ~4.3-4.5, not creep up.

**WHY SWING IMPROVES WITH swing_w=0:** load pos-error ||ep|| (load_w=10, in reward) and load vel-error
||ev||=swing are the SAME tracking quality -> can't hold position without matching velocity, so minimizing
load DRAGS swing down. Swing is carried by the LOAD term. This only kicked in once stall stopped
steamrolling load (400->50). Neither coord nor jerk is the swing lever -> the load term is.

===== HARD-WON LESSONS (the meta-rules, cost ~6 wasted runs) =====
* **EXPLORATION-TRAP:** any reward term on a quantity that EXPLORATION inflates (coord, jerk, action-
  smoothness) gets "minimized" by COLLAPSING EXPLORATION (shrink log_std), NOT by improving the policy ->
  reward looks better, deterministic metric stays flat, entropy dies. FIX = CLIP the per-step penalty
  (zero gradient above cap) so the only way to reduce it is to fix the MEAN. Calibrate such terms on their
  DETERMINISTIC magnitude but EXPECT sampled ~30x higher -> clip, don't just lower the weight.
* **REWARD BALANCE:** keep every term's typical contribution within ~1 ORDER OF MAGNITUDE. One over-weighted
  term (coord=50->-809k, stall=400) becomes the ONLY thing optimized -> everything else invisible +
  exploration collapses. Lean on penalty SHAPE (hinge/margin/quadratic/clip) for hard constraints, NOT raw
  weight. We hit this 3x (coord=50, stall=300/400, jerk=8-unclipped).
* **SAMPLED vs DETERMINISTIC:** team_ep_R (sampled sum over 2500x4 steps, exploration-laden) is NOT the
  metric -> DET_R (deterministic mean-per-step, held-out) is. A scary team_R (-350k) with a fine DET_R
  (-0.2) = an exploration-inflated term, not a bad policy. Best-checkpoint selects on DET_R (encodes ALL
  objectives; DET_loop is blind to stall), RESET per run (reward not comparable across schemes).
* **CHECKPOINT CLOBBER:** residual_mappo_last.pt is OVERWRITTEN on every Ctrl+C -> back up a good policy to
  a STABLE name (residual_mappo_prestall.pt) BEFORE stopping, and point WARMSTART there. Lost a good
  policy once this way. .pt files are in git -> `git checkout <hash> -- file.pt` recovers a checkpoint
  WITHOUT reverting code.

**CURRENT WORKING CONFIG (deep run, healthy 2026-08-01):** manifold_w=1, stall_w=50 (+stall_lin=1 hinge +
stall_margin=0.05 + stall_grace=20), load_w=10, swing_w=0, coord_w=0, jerk_w=8 (jerk_cap=0.05 CLIP),
ENT_COEF=0.003, STEPS_PER_ITER=20k, DOMAIN_RANDOMIZE=True, DET_R selection. All metrics trending right:
DET_R -0.20->-0.16, load 0.08->0.07, swing 0.21->0.19, stall vmin up / stall% down, jerk 0.065->0.062,
ent still slowly CLIMBING ~4.2->4.5 (drop ENT_COEF to 0.002 next run). Eval log cols: DET_R loop load vmin
stall% swing coord jerk. NEXT: let it finish; then DROP ENT_COEF to 0.002; then regime 4 (manifold_w down
to surpass expert). Trajectory-randomization phase later needs obs REFERENCE-
LOOKAHEAD + ~3-5x data + maybe bigger net (distill/surgery to keep learnings) + parallel collection (8 cores).

===== CANONICAL SUMMARY + CLEANUP TODO (2026-08-02) — the coherent story for the thesis =====
**REGIME TIMELINE (corrected):** (1) expert(manifold=1)+light stall(50)+load(10) -> first acceptable,
DET_loop ~0.18 on FIXED scenario; then warm-started DOMAIN RANDOMIZATION. (2) stall 50->300, swing-detour
tried+removed (swing_w=10 slowed formation->stall) -> deep stalls mostly gone. (3) stall 400+hinge+margin
(punish APPROACHING eps)+grace -- but 400 STEAMROLLED, SETTLED BACK to stall_w=50+hinge+margin (the KEPT
version). (4) the NO-OPs -> REMOVE: jerk (inert: clipped->constant->zero gradient), coord/Sum f_int (torque-
blind + degenerate min), swing_w (swing is STRUCTURAL, penalizing it just craters vmin), manifold-down
(marginal -0.166->-0.149, muddied, reverted).
**THE REAL WIN (not a reward tweak):** the TWO-HEAD SPLIT [dlam nullspace + dw range] broke the tracking-vs-
load Pareto see-saw. Plus per-drone GAE, clock in obs, time-indexed reward, blowup guard. KEEP all.
**KEY FINDINGS (the science):** (a) KEEP dw -- dlam-only made load AND swing WORSE (dw earns the load trim;
82-iter test: load 0.09-0.10 vs 0.078, swing rose to 0.27). (b) dw was MOONLIGHTING as a stall crutch
(subspace-clean but JOB-blended; nothing forced dlam=trajectory/dw=load). (c) **SWING IS STRUCTURAL** = the
cost of ANY aggressive correction (loops AND stall), because fixing the drone side disturbs the load through
the cables; base (no residual) has the LOWEST swing (0.10). THIS IS THE CEILING - physics, not a reward bug.
(d) PRIVILEGED CRITIC (delays->critic) expected ~NO GAIN: critic_loss was already ~0 (accurate baseline),
so little advantage-variance to remove; being tested now on two-head but temper expectations.
**CLEANUP TODO (strip back to the clean baseline, do together LATER - user deferred 2026-08-02):**
- ENT_COEF 0.0015 -> **0.0** (git-committed baseline; the 0.01/0.003/0.0015 tuning was a band-aid for the
  jerk exploration-trap; jerk being removed so the bonus's reason is gone. WATCH: 0.0 collapsed ent 4->-0.2
  under stall_w=400; with stall_w=50 + no jerk should hold, else re-add SMALL ~0.001).
- jerk_w -> 0 (remove the inert term), coord_w -> 0 (already), swing_w -> 0 (already), manifold_w stays 1.0.
- DISABLE_DW -> False (keep two-head). Net: back to regime-1-ish reward = manifold(1)+stall(50,hinge,margin)
  +load(10), NO entropy bonus, NO jerk/coord/swing. The clean, defensible baseline.
**CURRENT BEST (two-head):** held-out DET_R ~-0.15..-0.18, load 0.078, loop ~0.22, swing ~0.20, stalls ~gone.
See [[f2-residual-rl-plan]] UPDATE 2026-08-02 for the ablation details + trajectory-diversity ceiling plan.
