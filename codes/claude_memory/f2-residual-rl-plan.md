---
name: f2-residual-rl-plan
description: F2 = MAPPO residual RL. TWO-HEAD split [delta_lambda(null)+delta_wrench(range)] broke the tracking/load Pareto see-saw -> fixed-scenario DET_loop 0.18, load calm. Clock+time-indexed reward, 44-D obs, blowup guard. Now warm-started domain randomization.
metadata:
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-08-25T20:09:38.692Z
---

**F2 = MAPPO residual (delta_f) on the frozen analytic pR_dot base.** Long RL-debug arc,
latest turn 2026-07-29. delta_f* supervised warm-start RETIRED (base already tracks load).

**FAILURE (diagnosed, not jitter):** under desync the LOAD still tracks (~0.03-0.05 m) —
load = range(G), robust. Damage is entirely NULLSPACE (G.N=0): decentralized per-drone
lambdas disagree -> internal force distribution bad -> drones do slow large VELOCITY SWINGS
+ irregular loops, worse toward episode end; extreme spikes (v to 19-37) are TENSION
COLLAPSE (||f_i||->0 -> L0/T blows up v; the LLC filter bounds f_dot, so it's the L0/T
factor not f_dot). Load-error spikes are those blowups disturbing the load.

**KEY CONCEPTUAL CRUX (the clock):** "load fine" only means the drones are SOMEWHERE on the
load-serving (nullspace) manifold; infinitely many configs serve the load, desync put them
at an UGLY one, the expert loops are the NICE one. WHICH config is nice = a function of the
loiter PHASE. The residual obs lacked the clock -> it literally can't tell a good load-neutral
config from a bad one (observational ALIASING: same obs at different phases needs different
actions) -> best it could do was "near the loop band" ~0.5. The CLOCK is the zero-comms
COORDINATION signal (all drones share t -> implicitly agree on phase -> each tracks its own
clock-determined loop -> coordination emerges, no comms). Reward != obs: reward defines the
goal, obs is what the policy conditions on at decision time.

**REWARD EVOLUTION:**
- v1 MANIFOLD (phase-agnostic): r_i = -w_d*min_tau||p_i - p_i^central(tau)||^2 - w_v*relu(eps-||v_i||)^2
  - w_p*||e_p||^2. Fixed-scenario RL LEARNED but PLATEAUED ~0.5 (both caps). Demo: "learnt
  some stuff but still horrible esp toward the end, load still swings". Too permissive: accepts
  ANY point on the loop -> doesn't pin the phase (the coordination signal).
- v2 CURRENT = TIME-INDEXED: r_i = -w_d*||p_i - p_i^central(t)||^2 (the PHASE-CORRECT point at
  step t; env self._step -> expert_pos[i][idx]) + same stall floor + load guardrail. Forces the
  right point at the right time. NOTE: DET_loop metric changed (now phase-correct distance) ->
  starts ~1.3-1.4, NOT comparable to old 0.5; WATCH THE TREND. + CLOCK features in obs so it CAN.

**CLOCK IN OBS:** appended clock_features(t) (from collect_il_data; sin/cos over 7 freqs =14-D,
periodic/bounded = PHASE not raw t -> no memorization) to the residual obs, per drone at
t+clock_offset[i]. Residual obs now 44-D = load est(18)+own(6)+f_g(3)+f_lambda(3)+clock(14).
Same clock the base uses. NOT time-indexed with RAW t (would memorize).

**MAPPO/CTDE mechanics (what made it learn on the fixed scenario):**
- Shared residual Actor(44->3); centralized Critic(global true 42-D->scalar), training only.
- PER-DRONE reward + per-drone GAE (was TEAM-summed -> confounded credit 4x, KILLED learning;
  per-drone was a real fix). Critic target = mean per-drone return.
- LOG_STD_INIT=-1.0 (std~0.37; -0.5 exploration swamped the signal).
- DETERMINISTIC eval each EVAL_EVERY iters (eval_policy: mean action, no sampling) -> the HONEST
  DET_loop; sampled_loop is exploration-noise garbage. Deterministic eval is what turned "not
  learning" into visible learning.
- BEST-checkpoint save: residual_mappo.pt = best DET_loop (not last; last bounced). No critic saved.
- estimate_norm: obs normalization from a random rollout (44-D mixed scales; warm-start norm
  gone). DEVICE="cpu" (per-step GPU transfer was the 37->13s cost). REWARD_SCALE=0.01.
- DOMAIN_RANDOMIZE flag: FALSE now = FIXED scenario (FIXED_SEED=12345, FIXED_DELAYS=[1,2,2,1],
  deterministic env). "Learn one correctly FIRST" then True + big batch (STEPS_PER_ITER 15-25k)
  for generalization -- the first randomized run drowned in per-episode draw variance.

**CAP EXPERIMENT (done):** absolute-N vs proportional residual cap -> SAME ~0.5 floor -> NOT
authority-limited. Superseded by the two-head split below.

**TWO-HEAD SPLIT (current architecture, the big win):** the single force-residual delta_f had a
PARETO see-saw: tightening tracking (delta_lambda dir) injected a RANGE component that disturbed
the load; load_w=40 traded DET_loop 0.32->0.49. FIX = split the action by SUBSPACE:
  action = [delta_lambda (n=4), delta_wrench (6)]  -> act_dim 10.
  Applied in force space as df = slice_i( G+ @ delta_wrench + N @ delta_lambda ) using each drone's
  OWN local G+/N (since f_eff = G+(w_d+dw)+N(lam+dlam) = f_base + [G+ dw + N dlam]). So delta_lambda
  is load-NEUTRAL by construction (nullspace) -> reshapes trajectory; delta_wrench is a range TRIM
  on the already-PID'd w_d -> fixes load. Tracking now rides delta_lambda (conflict-free) and the
  load reward has its OWN actuator -> see-saw GONE. RESULT: fixed-scenario DET_loop 0.18 (< the
  0.32 the force-residual maxed at), load_w back to 10. Held-out seed (4242): DET_loop 0.28 /
  DET_load 0.12m -> strong generalization. finalize() untouched (base lambda rolls purely; residual
  reaches base only via sensed state). ORDERING DEBATE resolved: don't cascade (both leak under
  G-disagreement); JOINT two-head, because delta_lambda can't hurt load so tracking gets a free
  actuator. delta_wrench=classical-feedback-like (folds into w_d); works even w/o G-agreement bc
  it's FEEDBACK on the directly-sensed load error (base w_d already a PID; dw is a learned trim).

**CAPS (in Newtons, from measured scales w_d~6.9, f_base_i~1.72N, f_lam_i~1.85N):** cap_lam (frac
of ||lam_base||) and cap_w (frac of ||w_d||) are PROPORTIONAL. cap_lam=0.4 -> ~0.74N/drone null
force; cap_w=0.12 -> ~0.21N/drone load trim. Started tighter (0.25/0.05) after random exploration
BLEW UP the plant (tension collapse -> NaN -> SVD crash). Real fix = BLOWUP GUARD in env.step:
if state non-finite or drone speed>blowup_v(100) -> truncate cleanly + penalty (no crash, teaches
avoidance) -> so caps are for AUTHORITY not safety, loosened back to 0.4/0.12. mappo prints a
`blowups N` col. Blowups come from EXPLORATION SAMPLING (std .37 white noise ~cap mag every step),
NOT the mean (deterministic eval never blew). TODO-if-needed: small-init the actor head so residual
starts ~0 (pure base) -> gentler warm-up (not yet applied; user: "if it blows up too much").

**FILES:** expert_reference.py (->expert_ref.npz, per-drone expert path (N,T,3); RE-RUN if
get_reference_trajectory changes -- it's v_move=1.1, load to 11m; doc Figure_2 is older 7.2m).
residual_marl_env.py (44-D obs + clock + time-indexed reward + self._step + residual_cap_abs).
networks.py (+Critic). mappo.py (per-drone GAE, eval_policy, estimate_norm, best-save, knobs
ITERS/STEPS_PER_ITER/REWARD_SCALE/LOG_STD_INIT/EVAL_EVERY/DOMAIN_RANDOMIZE). demo_desync.py
(evals trained-vs-base on the EXACT training scenario; load_policy reads obs_dim from ckpt).

**BATCH FIX:** STEPS_PER_ITER 2500 (1 ep/update) -> bounced DET_loop 0.5<->0.9; raised to 10000
(~4 ep, collect() averages) -> stable monotone descent. For DOMAIN_RANDOMIZE need even more
(20000 ~8 ep) bc per-SCENARIO draw variance adds on top of sampling noise.

**CURRENT STATUS (2026-07-30):** two-head split PROVEN on fixed scenario (DET_loop 0.18, load calm,
see-saw broken). NOW running WARM-STARTED DOMAIN RANDOMIZATION: WARMSTART="residual_mappo_fixed.pt"
(actor+norm+critic), DOMAIN_RANDOMIZE=True (DELAY_CHOICES=(1,2) per drone + random noise seed),
STEPS_PER_ITER=20000, ITERS=150 (~3.7h). Checkpoint now SAVES CRITIC (critic_state key) + WARMSTART
loads it. eval_policy now on HELD-OUT EVAL_SEED=4242 (!= FIXED_SEED) and returns (loop, load) ->
prints DET_loop + DET_load. Backups: residual_mappo_fixed.pt (two-head fixed best), 
residual_mappo_forceresidual.pt (old 3-D). If randomization holds DET_loop/load across delays ->
DONE (generalizes). If it degrades -> broaden scope gradually / revisit. See [[f1-milestone]].

===== UPDATE 2026-08-02 (session findings + active experiments) =====
* **DELTA_W IS A STALL CRUTCH (ablation, ablate_heads.py):** ablation of the trained two-head policy
  (zero each head at eval) shows delta_w's biggest job is propping VMIN/anti-stall (r4: kill dw ->
  vmin 0.22->0.13, stall up), but dw acts through G+ (LOAD subspace) so every anti-stall nudge leaks
  into the load wrench -> SWING. And "base" (zero residual) has the LOWEST swing (0.10) vs trained
  ~0.21 -> **the RESIDUAL itself CAUSES swing** (loop-correction + dw stall-crutch); swing is not a
  dampable side-effect. => swing_w penalty CAN'T work (proven twice: swing flat, vmin craters). The
  two-head split is clean by SUBSPACE (dlam load-neutral, dw in range) but NOT by JOB (nothing forced
  dlam=trajectory/dw=load; policy blended them, dw moonlights on stall).
* **ACTIVE EXPERIMENT: DELTA_LAMBDA-ONLY.** env flag `disable_dw` (zeros dw slice in step) + mappo
  `DISABLE_DW=True`. Warm from residual_mappo.pt(~r4), manifold_w back to 1.0, swing_w=0. Tests: can
  dlam alone cover loops+stall load-neutrally (swing -> base ~0.10) leaving load to base PID? If load
  holds -> drop dw. If load degrades -> keep dw but add light ||dw|| penalty to FORCE the clean split
  (dw only when it helps load). NOTE dlam-only runs HIGHER-GAIN: sampled_loop >> DET loop (0.355 vs
  0.25, flipped from two-head where sampled<=det) -> LOG_STD -1.0 is a bit hot for it (lower to ~-1.3
  if descent slow). Backups: residual_mappo_r4base.pt = pristine -0.149 two-head (last clean copy).
* **PRIVILEGED CRITIC (ADDED, ready to run):** asymmetric actor-critic. `critic_input(env)` = state(42)
  + per-drone delays(n), 0-centered; state_dim=42+n; warmstart critic REINITs on dim mismatch (guarded).
  Actor still local obs -> decentralization intact (critic training-only). WHY: delays are a FIXED,
  KNOWN per-episode scenario param the critic couldn't see -> it baselined the AVERAGE -> scenario luck
  leaked into the advantage as noise (misattributed to the action) = the "best@iter1 then drift" of every
  fine-tune. Feeding delays lets V explain-away scenario difficulty -> advantage isolates the action.
  TELL IT WORKS: critic_loss rises off ~0.000 + drift eases. (Noise variance is IRREDUCIBLE - future
  draws don't exist yet - so it needs MORE DATA, not the critic; delays are the reducible part.)
* **TODO (LATER, after privileged-critic test): MULTI-SCENARIO EVAL.** DET_R is currently ONE hand-picked
  held-out case (EVAL_SEED=4242, EVAL_DELAYS=[1,2,2,1]) -> a possibly-biased single stick (repeatable but
  not representative). Make eval_policy AVERAGE over a small fixed list of seed x delay combos -> low-var
  estimate of true generalization. Do before trusting any absolute "is it actually better?" call.
* **CEILING = ONE TRAJECTORY (strategic).** All training is on ONE loiter reference (+noise/delay). More
  STEPS_PER_ITER only adds noise DRAWS, not trajectory diversity -> can't teach the general disturbance-
  rejection law -> memorizes one loop = a ceiling. RAISE it with TRAJECTORY RANDOMIZATION. Re lookahead:
  the CURRENT-instant reference is ALREADY in obs implicitly via base forces f_g/f_lambda (base controller
  computes them from ref pos+vel) -> the residual rides a base that knows the trajectory -> try multi-traj
  with CURRENT obs FIRST; add reference-LOOKAHEAD only if anticipation-limited lag shows on curvy parts.
  (The clock is FAST sin/cos basis, NOT a memorizable time index - policy localizes via physical state.)
* **COORD term post-mortem (why ||sum f_int||^2 did nothing):** it penalizes only NET internal FORCE, is
  TORQUE-BLIND (misses Sum r_i x f_i) -> a pure-twist leak passes through; the leak is also tiny/spike-
  dominated (median 0.03N); and it minimized DEGENERATELY (drones drift off-loop to cancel forces). Right
  quantity if ever revisited = FULL wrench leak ||G_true f_applied - w_desired||^2. But not the swing lever.

===== GENERALIZATION PHASE (2026-08-03) — one trajectory was the toy; now MANY =====
**DECISION: generalize over PAPER-STYLE rest-to-rest QUINTIC pose trajectories** (p2025ut Fig.10 family),
NOT piecewise turns, NOT aggressive constant-speed arcs. Rationale from examine_base.py (new file: runs the
NOISE-FREE base on candidate trajectories, patches controller.get_reference_trajectory, reports load-track err):
  quintic_1d 0.010 / quintic_3d 0.017 (mean err, m) == paper's mean||ep||=0.015; NO accel feedforward needed.
  vs harsh arc_r2 0.20 (constant-speed 1.1m/s -> centripetal accel ~0.6 -> PID lag; THIS is why feedforward
  first seemed needed - ARTIFACT of too-harsh test). vs current toy straight 0.049 (velocity-STEP spikes 0.39).
  => gentle quintics (peak accel ~0.06 for 1m/10s) are EASIER on the base than our current toy. Base is READY
  for smooth 3D as-is. (Accel-feedforward = return a_Ld + Ka term = only a nice-to-have for FAST maneuvers later.)
**PAPER Fig.10 = rest-to-rest quintic 6D POSE:** x,y,z each move 1m in 10s + pitch 10deg + roll 5deg (yaw~0),
then hold 5s; eps=0.55. CRUCIAL: paper Table I gains == our controller.py EXACTLY (Kp5 Kv2 Ki0.9 KR0.5 Kw0.06
KiR0.1) -> the paper's base IS our base -> Fig.10's 0.015m is directly what we should get. Base ALREADY does
orientation (error_calculation eR/ew + wrench_controller KR/Kw/KiR); today get_reference_trajectory returns
R_Ld=I always -> full 6D just needs R_Ld set to a roll/pitch quintic.
**BUILD ORDER:** (1) parameterize get_reference_trajectory -> quintic pose from params (per-axis deltas, ramp,
hold), env-set per episode. (2) LIBRARY generator: K sampled trajs -> run noise-free base (expert_reference
loop) -> expert_lib.npz (K x (N,T,3) + params); hold out M for eval. (3) per-episode sampling into env +
held-out eval set (finally kills the single-biased-stick DET_R). (4) obs: try CURRENT obs first (base forces
f_g/f_lambda carry the current-instant reference -> residual rides a base that knows the traj); add ref-
LOOKAHEAD only if anticipation lag. Do 3D POSITION first (R_Ld=I), then add roll/pitch for full 6D.
**BOTH F1 AND F2 RETRAIN ON THE LIBRARY** (user flag 2026-08-03): F1 = the IL-learned DECENTRALIZED base is
currently SINGLE-trajectory -> retrain its data collection (collect_il_data etc.) over the library too, else
the residual rides a base that only knows one loop. Stack generalizes together: F1 base -> F2 residual.
NOTE current get_reference_trajectory is a piecewise straight move (hold5 -> +x@1.1 to 11m -> stop), T_END=25.

**INFRA/SCALING (decided 2026-08-03):** parallelization is a DESIGN CONSTRAINT, not a later add-on (K-traj is
too slow serial). User moving to UBUNTU + SLURM. => (1) trajectory is an ENV-INSTANCE attribute threaded down
compute_forces->error_calculation, NOT a module global (global is per-process + fork-inherited -> silent
cross-contamination with multiple envs/workers). (2) parallel LIBRARY gen first (embarrassingly parallel,
offline, low-risk) then worker-pool RL collection (CPU workers collect + broadcast actor weights each iter;
GPU learner for the PPO update once nets grow). (3) OS-agnostic, per-worker RNG seeding (base+rank), worker
count from $SLURM_CPUS_PER_TASK, robust checkpoint/resume for walltime limits. Build+validate single-process
on a TINY library first, THEN turn on parallel RL collection (don't debug traj-correctness + MP at once).

**BOTTLENECK DIAGNOSIS -- TODO: a DIAGNOSTIC RL actor sweep (bigger + more-info + memory) to find what caps
F2, to inform the MAIN actor design.** Long epistemic thread (2026-08-03), conclusions:
- F2 SPECIFICALLY fights decentralization/desync. Clean/undelayed state = DEGENERATE (falls back to F1, deletes
  the problem) -> useless as an experiment or target.
- It's a POMDP: the missing info is the CURRENT true state (each drone's load estimate is DELAYED + NOISY).
  Delay-value is only PARTIAL info (tells you staleness, not the state). The current state is RECOVERABLE only
  by PREDICTION FROM HISTORY -> so MEMORY (RNN/GRU / stacked frames) is a principal lever, NOT width. Could be
  BOTH memory AND capacity (the coord feedback law may be genuinely complex) -- we DON'T know, don't guess.
- NO SUPERVISED SHORTCUT to split representability vs optimization: there is no ground-truth optimal target.
  The delta_f* supervised warm-start FAILED for this exact reason (base already tracks load -> target trivial/
  redundant, encoded nothing). Distilling the current F2 = CIRCULAR. And an ORACLE (privileged-info) actor is
  NOT a valid imitation/distillation target: in a POMDP the more-informed-optimal policy is STRUCTURALLY a
  different mapping (conditions on the TRUE STATE, acts confidently) from the partial-info-optimal one (must
  condition on BELIEF and HEDGE). Imitating it is infeasible AND wrong.
- So an oracle's ONLY valid use = a PERFORMANCE UPPER BOUND for ATTRIBUTION (e.g. comms-oracle = give others'
  true states, delay-oracle) -> a NUMBER telling you IF and WHICH missing info costs reward, never a teacher.
- => the too-small-vs-needs-memory-vs-both question has NO clean shortcut; answer it by an ARCH SWEEP IN RL
  {MLP, +GRU/history, wider} comparing ceilings (optimization-confounded, EXPENSIVE -> this is WHY parallel
  compute matters). The multi-traj TRAIN-traj vs HELD-OUT-traj gap is the one free signal: both-bad-&-close =
  underfit, train-good/held-out-bad = overfit (can't split memory-vs-width by itself, but separates fit regime).
  Design the diagnostic sweep to vary {info (own-history, delay-val, comms-oracle-as-bound), capacity, memory}.

===== F1-BASE RE-WIRE (2026-08-18) — F2 base updated to the new LOG-TAP F1 net =====
F1 shipped the log-spaced lambda-history (66-D input, (256,256) net) -- see [[f1-phase-slip-lambda-history]].
The F2 `LocalModelAgent` (the FROZEN F1 replica that produces the base `lams` in env.step -- NOT the RL
actor/critic, those are untouched) had gone stale against it. Three fixes in `residual_marl_env.py`:
(1) env `Actor(...)` now reads `hidden` from the ckpt (was hardcoded (128,128) -> couldn't even load the
256-wide net); (2) `LocalModelAgent` now carries the full-depth log-tap buffer (`init_lam_history`/`push_lam`,
`build_input(t,vR,self.lam_buf)`) -> 66-D input (was feeding 1-D prev_lam -> 30-D, dim-mismatch crash);
(3) added the in-loop EMA (`LAM_LP_TAU=0.01`) in `finalize` -- the net was TRAINED against the EMA'd feedback,
F2 had none; now the APPLIED (EMA'd) lambda drives force AND rolls prev_lam+lam_buf, matching deploy_prdot.
Also `sanity_f2.py` zero-action fixed from stale 3-D to the 10-D [dlam(n),dw(6)]. RESULT: sanity_f2 (zero
noise/residual) reproduces deploy_prdot: mean-track 0.0814 vs 0.0813, max 0.3868 == identical. Drone speed
means ~1.29-1.35 vs deploy 1.40-1.43 = the pre-existing reconstruct_lp(noise-robust) vs analytic recon diff,
lives in the load-NEUTRAL nullspace so tracking is untouched (benign). F2 machinery GREEN on the new base.
===== EXPERT LIBRARY + DEMO REWORK (2026-08-18) — the per-traj "dictionary" for multi-traj RL =====
`expert_reference.py` generalized to the shared primitive: `expert_path(traj, t_end)` = one noise-free
CENTRAL ClassicalAgent (CasADi) rollout -> per-drone (dpos,dvel,load); the IDEAL the reward tracks.
`build_library()` -> `expert_lib.npz` = the per-trajectory DICTIONARY (ragged, keyed `{set}__{idx}`,
manifest meta_set/label/tend/key; `load_library()` reloads) over the RL-training set (default +
custom_set(5) + train_set(N_TRAJ=50) = SAME 56 collect_prdot uses -> F1 base + F2 reward share trajs)
PLUS showcase short/long for demos. Run offline: `python expert_reference.py lib`. Back-compat single
`expert_ref.npz` still via `main()`. compute_forces already takes traj -> no controller change. THIS is
the "generalize F2 over many trajectories" prerequisite: NEXT (RL files) = env reward reads its per-episode
expert path from the lib (env.expert_pos set per episode by mappo, keyed to the sampled traj index) instead
of the single hardcoded expert_ref.npz; mappo samples (traj, expert_ref) pairs from the same generator.
demo_desync.py REWORKED: 2 modes (PURE = RL-under-noise vs ideal-no-noise from expert_path live; NOISY =
RL-under-noise vs base-controller-under-noise, cooked), SHOWCASE_KIND/IDX one-at-a-time, faded-overlay XY+vel
kept (user wanted the OG plots) + a SEPARATE load-XY figure (curiosity), DRONE XY is the headline (load rides
range(G) so load-tracking hides the failure). Verified: NOISY/long/line RL held load 0.093m mean under
held-out seed 8888 delays [2,1,1,2] (base controller 0.079m -> load fine either way, the failure is the drones).

===== MULTI-TRAJ RL WIRED (2026-08-21) — mappo now trains over the trajectory library =====
expert_lib.npz BUILT (64 refs: train=56 [default+custom_set(5)+train_set(50)] + showcase_short(4) +
showcase_long(4)). `training_pairs()` -> 56 index-aligned (traj, expert_dpos) pairs (trajs rebuilt from the
SAME generators the lib used; asserts length). mappo.py wired: `TRAJ_RANDOMIZE=True` -> collect() samples a
(env.traj, env.expert_pos) pair PER EPISODE (alongside the existing per-episode delay/noise randomization);
env needed NO change (already threads self.traj + reads self.expert_pos in the reward). eval_policy/estimate_norm
PIN back to the default traj (env.traj=None + env.default_expert_pos captured in main) so DET_R stays a
consistent cross-iter metric even with traj-randomization ON. CORRECTNESS FIX: mappo env was created at the
default end_time=25 but trajs+experts are 35s -> episodes truncated at 25s, missing the 20-30s move-ends +
the 25-35s zone F1 was built for; fixed to `end_time=T_END`(35) -> 3500-step episodes. Smoke-tested: collect
3500 steps finite no-blowup, eval_policy OK. NOTE r4base warmstart was trained 25s/single-traj -> DET shifts
(fresh baseline expected); RETRAIN. TODOs: (1) held-out-TRAJ eval (showcase is in the lib, held out from train)
for the real generalization signal — currently eval is default-traj (in-distribution, comparable); (2) the
trajectory is now a hidden per-episode scenario param the critic can't see (like delays were) -> consider
privileged-critic traj-conditioning if advantage looks noisy.
+ REFINEMENTS (2026-08-21): (a) ANCHOR GUARANTEE like F1's collect: `training_pairs()` now returns
(pairs, n_anchor=1+len(custom_set())=6) with the NON-quintic anchors (default line + solver-engaging
customs) FIRST; mappo `GUARANTEE_ANCHOR=1` forces the first N episodes/iter to draw from pairs[:n_anchor]
(bump to 2 for two), rest from all 56 -> every iter sees a non-quintic. (b) PER-STEP DELAY (was fixed
per episode -> unrealistic): env now walks `self._delay_cur` as a bounded random walk in [0, ctrl_delay_i]
(+-1/step, advanced in _update_estimates, seeded start in reset) -> a drone falls behind then catches up.
ctrl_delay stays the per-episode MAX/nominal = the critic's privileged scenario param (unchanged); the
per-step jitter is unobserved noise the policy must be robust to (like the sensing noise draws). Verified.
NB the env NEVER samples trajs/scenarios itself -- mappo.collect drives them by setting env.traj/expert_pos/
ctrl_delay before reset(); expert_ref="expert_ref.npz" in the env is just the default baseline.

===== REGIME-1 CURRICULUM REBUILD + RUN-READY CLARIFICATIONS (2026-08-21) =====
Decision: REBUILD the working reward curriculum on the new multi-traj base, don't jump to regime 3. The
current reward was bloated with regime-4 stuff that DIDN'T work (jerk/coord + manifold tune-down). The
curriculum that WORKED (user's github/history): R1 = expert/manifold + LIGHT stall(50) + load -> first
acceptable (~0.18 loop); R2 = stall 50->300 (swing 'detour' removed); R3 = stall 400 + HINGE + MARGIN
(punish approaching eps) = good; R4 (jerk + ||sum f_int||^2 + manifold down) = DIDN'T WORK.
SET NOW = pure R1 in residual_marl_env defaults: manifold_w=1.0, load_w=10, stall_w=50, stall_lin=0,
stall_margin=0 (plain relu(eps-v)^2 floor), swing_w=0, jerk_w=0, coord_w=0, stall_grace=20. KNOBS to step:
R2 -> stall_w=300; R3 -> stall_w=400 + stall_lin=1.0 (hinge) + stall_margin~0.05 (margin). manifold stays 1.0.
ENT_COEF -> 0.0 (was 0.0015): the entropy COLLAPSE (ent 4->-0.2) that forced ENT_COEF>0 was a REGIME-4
JERK artifact (jerk penalty cheapest to cut by shrinking log_std). No jerk -> nothing pushes entropy down
-> 0 works (PPO clip + Gaussian LOG_STD_INIT=-1.0 natural exploration). If entropy RUNS AWAY at 0, unexpected.
WARMSTART=r4base KEPT (user's call). log_std REINIT on warmstart (actor.log_std.fill_(LOG_STD_INIT)) is a
FEATURE not a bug -> r4base is a regime-4 net with a COLLAPSED log_std; reinit restores healthy exploration
(without it you'd resume dead exploration). NB warmstart warms ACTOR only -> the CRITIC REINITs here
(r4base critic predates the privileged-delays state_dim 46) -> warm-actor/fresh-critic -> ignore the first
few DET_R/critic_loss (value re-learn dip).

CODE CLARIFICATIONS (verified, so we don't re-litigate):
- eval/monitor metrics load_err, load_verr(swing), min_speed, coord, jerk are computed in step() from the
  TRUE plant state (obs42/npos/nvel/v_cur), NOT any drone's noisy/delayed estimate, and broadcast IDENTICALLY
  into EVERY agent's info -> infos[agents[0]][...] is just ONE COPY of the global truth (min_speed = global
  min over all drones, not agent 0's). loop_dist/lambda/prdot_own ARE per-drone. CRUCIAL: `infos` is
  logging/eval ONLY -- the ACTOR conditions on `obs` (local, noisy, delayed, from _build_obs), never infos ->
  no privilege leak (coord etc. are fine in infos). eval SHOULD use true global state to score real perf.
- caps cap_lam=0.4 (frac ||lam_base||) + cap_w=0.2 (frac ||w_d||) are PROPORTIONAL; act_space Box(-1,1) is the
  nominal Gaussian range but the real limiter is _clip_norm scaling each head to the cap NORM (not per-comp clip).
- CRITIC input is NOT normalized (only the actor gets (obs-om)/os_); critic_input(state42+delays) is raw,
  same mixed scale -> a real OVERSIGHT/gap (critic would benefit from its own state_mean/std), NOT principled.
  Left OFF for regime-1 parity with the working setup; add later if the value fit looks poor. [optional]
EVAL = 2-TRAJ MEAN (DONE 2026-08-21, closes the held-out-eval TODO): best-net SELECTION was a single lucky
stick (default traj). Now `eval_scenarios()` -> [("line", None, default expert), ("long_quintic1", showcase
held-out quintic + its lib expert)]; `eval_policy(...,scenarios)` loops both (same EVAL_SEED/EVAL_DELAYS),
selects on the MEAN DET_R (worst-case for vmin/loadmax/stallfrac), prints per-traj loop spread. EVAL_EVERY
2->4 so the 2-traj mean costs the SAME total as the old 1-traj eval. Line is in-dist anchor; quintic is
held-out generalization. VALIDATED: r4base scores line loop 0.22 (its trained traj) but held-out quintic
1.21 -> mean 0.72; a line-only eval would've mis-saved it as 0.22. Add a 2nd held-out quintic later if 2 is
too noisy. STILL-OPEN: (1) privileged-critic TRAJECTORY-conditioning (traj is a hidden per-episode scenario
param like delays) if advantage looks noisy. (2) [optional] critic input normalization (only actor is normed).
RUN-READY 2026-08-21: all checks pass (syntax, config, expert_lib 56/6, regime-1 weights pure, warmstart
44->10 loads). expected: fresh DET baseline (r4base was 25s/single-traj/regime-4), target ~0.18 loop before R2.

===== FIRST MULTI-TRAJ RUN DIAGNOSIS + CRITIC-TARGET FIX (2026-08-21) — the flagged TODO bit =====
Ran regime-1 multi-traj (warmstart r4base). 40 iters SYMPTOM: `line` DET loop FROZEN at 0.22-0.23 (== the
F1 BASE alone -> residual doing ~nothing in-dist), `long` quintic bounced 0.95-1.2 no trend, DET_R crept
-1.27->-0.95(it12) then DRIFTED BACK, blowups 1-3/iter persistent, team_ep_R swinging -9.6k..-47k. Entropy
4.16->3.90 = HEALTHY gentle sharpening (NOT the problem; red herring). ROOT CAUSE = the flagged privileged-
critic TRAJECTORY-conditioning gap, now BLOCKING: the reward is dominated by `manifold_w*||expert_pos[i][idx]
- p_i||^2` but `critic_input` = state42+delays could NOT see the target. SINGLE-traj (how r4base was made)
the target is a fixed fn of load phase (IN the state) -> V memorized it. MULTI-traj: the SAME load config maps
to 56 DIFFERENT targets -> V(state) can't tell which traj -> value baseline COLLAPSES -> advantage is swamped
by the traj DRAW difficulty (that -9.6k..-47k spread), not the action -> PPO gradient thrashes -> actor drifts,
line pinned at base, blowups never learned away. FIX (mappo.py `critic_input`): append the per-drone TIME-
INDEXED expert target `expert_pos[:,idx,:].flatten()` (3n=12-D) -> critic 42+n+3n = 58-D. V can now difference
target vs the drone positions already in state42 -> value is traj-aware -> advantage isolates the ACTION.
Training-only privileged -> valid CTDE (actor untouched, still local 44-D). state_dim now derives from
critic_input(env).shape[0] (robust). Critic REINITs from warmstart (was already reiniting). Smoke-tested 58-D.
EXPECT on rerun: DET_R climbs (not just it-1 luck), line loop drops below 0.22, blowups decline as the policy
finally gets a real avoid-signal. If blowups still won't -> lower LOG_STD_INIT slightly (they never hit 0 at
std .37). This CLOSES the "privileged-critic trajectory-conditioning" TODO. Remaining optional: critic-input
normalization (only actor is normed). THEN step R2 (stall_w=300) once line+quintic both settle ~0.18.

===== CRITIC-TARGET FIX RESULT + BLOWUP HANDLING (2026-08-22) =====
CRITIC-TARGET FIX WORKED but only MODESTLY. 150-iter run (warmstart r4base, critic 58-D): DET_R -1.24 ->
~-0.3, held-out quintic loop 1.13 -> ~0.48, LINE loop FLAT ~0.22 the whole time. So ALL the gain is on the
quintics (base already near floor on straight lines); line-flat is EXPECTED/fine. BUT real-terms modest: the
reward (DET_R) moved more than tracking because it folds load/stall; the TRACKING metric (loop) only went
0.68->0.35 mean, and quintic ~0.48m off-phase is VISIBLY mediocre in demo_desync. User: "improved a bit, not
5x... not following trajectories well even ignoring blowups". OPEN: tracking QUALITY is the next problem.
BLOWUPS: happen in DETERMINISTIC eval too (user corrected me -- I'd wrongly claimed the mean policy is safe;
it is NOT, this is a DEPLOYMENT hole). Two fixes shipped:
(1) eval_policy MASKING BUG FIXED (mappo.py): on blowup env sets agents=[] + returns ONE info row (loop_dist=
    load_err=5.0, min_speed=0, blowup=True) -> a LATE blowup was ~invisible in the MEAN (one -100 among 1000s
    of steps -> DET_R barely moved; truncation even makes the avg look BETTER) -> a blowing net could be SAVED
    as "best". Now eval_policy detects `infos[a0].get("blowup")`, DISQUALIFIES that traj (sentinel reward -50,
    loop 5.0) so best-net selection can never pick a blowing net, counts n_blow, returns out["blowups"],
    prints "DET_BLOWUPS N". Also guards empty `speeds` (blowup before stall_grace) -> vmin 0/stallfrac 1.
(2) SPEED-CEILING reward term ADDED (residual_marl_env.py) -- user approved "punish high velocities". The
    blowup = TENSION COLLAPSE -> v explodes 19->37->guard@100, but the ONLY signal was the SPARSE TERMINAL
    -blowup_penalty(100) -> no gradient on the APPROACH -> mean policy never learned MARGIN. New DENSE term:
    overspeed_w=10 * min((v-overspeed_v)^2, overspeed_cap), overspeed_v=8 (>> cruise ~1.4 / aggressive <~4,
    << collapse 19-37 -> ZERO in normal ops, can't disturb tracking), overspeed_cap=30 (caps per-step (v-8)^2
    like jerk_cap -> dense gradient in the RECOVERABLE band v~8-13.5, no giant advantage spike once past
    saving). Wired: __init__ params + self.overspeed_* + reward loop `over2` term. WHY NOT just bump
    blowup_penalty: a bigger SPARSE TERMINAL spike injects huge advantage variance (F1 gradient-dominance
    failure mode) + bluntly collapses the residual toward 0/base (kills tracking) -- dense pre-cliff shaping is
    the right lever. CAVEAT told to user: speed term fixes SAFETY (divergence), NOT tracking quality.
NEXT-LEVER for TRACKING (recommended, not yet done): before any arch change, a CHEAP diagnostic to localize the
ceiling -- (a) log residual CAP SATURATION (||dlam||/cap_lam, ||dw||/cap_w) in DET eval: pinned=authority-
limited -> raise cap_lam; not pinned -> representation/optimization; (b) compare quintic loop RL-under-noise vs
BASE-under-noise (demo NOISY) -> is the residual even helping / is it a base problem. Only THEN consider
manifold_w up (1.0 is deliberately the "proven" value) or the flagged POMDP MEMORY arch (GRU/frame-stack --
desync is a POMDP, memoryless MLP has a ceiling; see the 2026-08-03 arch-sweep note above).
CONTINUE-TRAINING (asked 2026-08-22): main saves BOTH residual_mappo.pt (best, mid-run) + residual_mappo_last.pt
(last, at end/Ctrl-C), both FULLY resumable (actor+critic+norm+best_reward). Resume via WARMSTART=_last when the
run was still CLIMBING (this run was -> _last is at the frontier w/ the matching critic); the "prefer best" rule
only applies when DRIFTING (_last past peak). Resume reinits log_std to LOG_STD_INIT + drops Adam moments (fine).
best_reward RESETS each run; DET_R NOT comparable across the reward-scheme change (overspeed added) though critic
loads clean (58-D unchanged).

===== TRACKING CEILING = AUTHORITY-LIMITED, not capacity (diagnose_f2.py, 2026-08-22) =====
NEW TOOL `diagnose_f2.py` (standalone, loads a trained ckpt, NO training loop): runs DET eval on
3 scenarios [line, a TRAINING quintic pairs[n_anchor+5], the HELD-OUT long_quintic1] under the same
EVAL_SEED/EVAL_DELAYS, reports loop + residual CAP SATURATION (sat = raw head norm / cap; >=1 => clipped).
Needed env plumbing: step() now computes self._sat_lam/_sat_w per drone and puts "sat_lam"/"sat_w" in info
(diagnostic-only; actor never sees info). Run: `python diagnose_f2.py [ckpt]`.
FINDING on residual_mappo.pt (best of the critic-target run): line loop 0.220 / train_q 0.309 / held-out 0.345.
=> (1) NOT a fit-regime problem: train vs held-out gap +0.036 (tiny) -> GENERALIZES fine -> bigger net / more
data / regularization NOT indicated (closes the "maybe bigger net?" fork -- 128x128 is not the bottleneck).
(2) AUTHORITY-LIMITED (smoking gun): the tracking head dlam is CLIPPED 60% (line) / 86-87% (quintics) of steps,
raw output 1.1-1.5x cap_lam -> the policy WANTS more tracking authority than cap_lam=0.4 allowed. Load head dw
has headroom (sat ~0.43, clip <1%) -> base handles load, as expected. So the ~0.3 plateau = "not ALLOWED to
apply", not "can't learn". FIX = RAISE cap_lam (cheap, targeted), NOT grow the net. The overspeed term is the
guardrail that makes raising cap_lam safe (more authority = more tension-collapse risk = more blowups).
ACTION TAKEN (user, 2026-08-22): cap_lam 0.4 -> 0.8 (2x). overspeed_v 8 -> 4 (user: "any higher and it's
already not right" -- caught divergence too late at 8; normal ops stay < 4). Rerunning mappo.
WATCH (interaction): cap_lam^ and overspeed_v v push OPPOSITE ways (more authority = faster corrections; lower
ceiling penalizes them earlier). If quintic loop doesn't improve as much as un-clipping predicts, the overspeed
term at v=4 may be biting the unlocked authority. TELL via diagnose_f2: clip_lam should DROP (authority freed);
if loop stalls with speeds pinned ~4 -> raise overspeed_v or trim overspeed_w. vmin=0 in diagnose_f2 is the t=0
startup artifact (doesn't skip stall_grace) -- benign. If still saturated at 0.8 -> raise cap_lam again; only if
loop unsatisfying with saturation CLEARED do net/memory (POMDP frame-stack) become the question.

===== THE COORDINATION FRONTIER + ZERO-COMMS PUSH (2026-08-23) — the real F2 wall =====
3-WAY DECOMPOSITION (diagnose_f2.py, gt1 = best net): base_CLEAN loop 0.013 (F1 tracks quintics ~PERFECTLY,
no desync) -> base_NOISY loop 1.10 / load 0.036 / coord 0.047 (desync = CATASTROPHE, +1.09) -> RESIDUAL loop
0.33 / load 0.086 / coord 0.352. So the residual RECOVERS ~70% of a 1.1m catastrophe (NOT a failure) but the
LEAK (coord=||sum f_int||) is 7.5x the base's, and load-hurt is 2.4x (0.036->0.086). BASE IS EXONERATED (0.013
clean) -> NOT an F1 problem. The wall is DECENTRALIZED COORDINATION (a POMDP).
WHY THE LEAK IS 7.5x UNDER RL (user's model, correct; I conceded "bigger" was wrong -- dλ is CAPPED 0.5x):
base runs the IDENTICAL F1 net on the SHARED load view -> all drones hold AGREEING λ -> nullspace forces cancel
-> only tiny G-disagreement leak (0.047). RL adds per-drone dλ conditioned on each drone's OWN state (different
per drone) pushing toward each drone's OWN target -> MISALIGNED λ (not bigger, capped) -> forces stop cancelling
-> 7.5x leak. It's MISALIGNMENT, not magnitude.
WHY THE LOAD SURVIVES while trajectories drift to 1.1m (the paradox): (1) the drift IS a NULLSPACE motion (G·N=0
-> zero wrench) -- huge in config space, ~0 in wrench space. (2) load DOF = REGULATED closed loop (w_d PID on
directly-sensed load error, range space) -> rejects the bounded leak -> stays 0.036/0.086. Internal-config DOF =
UNREGULATED -> small per-step misalignment INTEGRATES into unbounded drift. (3) FORCES are CAPPED (bounded leak,
bounded load) but STATES are the unbounded TIME-INTEGRAL of the bounded misaligned force (velocity ramps, position
double-ramps). Load feels bounded FORCE; trajectories ARE the unbounded integral. MAGNITUDE vs FREQUENCY settled by
measurement: coord ratio RL/base = 7.5x but load ratio = 2.4x -> load loop is SUBLINEAR (rejects most of the bigger
leak; extra leak is oscillatory/high-freq the load inertia filters). PID does NOT handle it worse -- input is 7.5x.
WHY THE OLD coord_w PENALTY FAILED = INCENTIVE WITHOUT ABILITY: penalizing ||sum f_int|| with no info to coordinate
-> only reachable minimum is DEGENERATE (stop pushing / drift off-loop to cancel) -> kills tracking. Reward alone
can't fix a coordination problem the policy has no information to solve.
LITERATURE (user, 2026-08-23): paper A used history+goal-traj -> "load history didn't help much", 0 load misalign
but tracking SUCKED = the OPPOSITE frontier corner (0 leak, bad tracking; we're good-tracking/some-leak). => history
ALONE does NOT break the frontier. paper B "had to use relative drone positions" = INTER-DRONE info breaks it.
KEY INSIGHT why history fails here: the load is SO well-regulated (0.036) it barely reveals the misalignment -> the
shared observable memory would exploit is nearly SILENT. Only reliably-shared coordination signal = the CLOCK.
USER DECISION: push the frontier WITHOUT inter-drone comms (no easy way out). => CTDE zero-comms-at-execution levers.
EXPERIMENT RUNNING (2026-08-23) -- "reward(incentive) + obs(ability)":
(1) OBS +18: DESIRED LOAD STATE [p_d,R_d,v_d,ω_d] from get_reference_trajectory(t,traj), the SHARED reference (same
    for all drones -> target info WITHOUT inter-drone data). residual obs 44 -> 62. _build_obs + _obs_space(48+clock).
(2) REWARD: leak_w * ||G_true @ f_int||^2 -- FULL-WRENCH (force+TORQUE) leak of the applied INTERNAL forces via the
    TRUE grasp. Torque-aware fix for the old force-only coord_w; ISOLATES the nullspace (built from f_int, doesn't
    touch dw range). CTDE-legal (train-time global reward, local execution). leak_w=1.0 (~manifold_w; history:
    coordination terms STEAMROLL tracking if >> manifold -- coord_w=50 & overspeed_w=10 both cooked it). env now
    also stores self._f_int + infos["leak"]. NEEDS: from optimizer import calculate_grasp_and_nullspace; from
    controller import get_reference_trajectory.
(3) NET widened (128,128) -> (256,128,64) FUNNEL (mappo HIDDEN, threaded to Actor+Critic), COLD START (WARMSTART=None,
    obs+net both changed -> nothing loads; clean read, no inherited bad-coordination basin). NOTE old 44-D nets can't
    run in this env now (obs mismatch) -- but gt1 baseline recorded above, compare new-net diagnose_f2 vs it.
SUCCESS = diagnose_f2 shows coord/leak DROP toward base (~0.047) while loop HOLDS/improves (coordination emerges,
zero comms). FAILURE = loop DEGRADES to buy the coord drop -> sliding the frontier (leak_w too high) OR shared-info
genuinely insufficient -> then it takes inter-drone info (relative positions) = confirmed by comms-oracle upper bound
(still-untested: give each drone others' true states, retrain -> if coord/loop/load all drop, info is the bottleneck).
NEXT ARCH IDEAS IF THIS PLATEAUS: split residual into a COORDINATED head (shared obs only -> identical across drones
-> cancels by construction, low leak) + small capped PRIVATE head (own-state, own-tracking); or recurrent belief-state
(fights the silent load channel -> modest, per paper A). diagnose_f2 3-way + coord + cap-saturation is the read tool.

===== COORDINATION-RUN TUNING + NORM BUG + INFRA (2026-08-23, ACTIVE — user compacting, will circle back) =====
LEAK_W MIS-SCALE (fixed): leak_w=1.0 STEAMROLLED -> team_ep_R -539k, critic_loss 225, coord RISING 0.13->0.38,
everything degrading. Full-wrench leak is MUCH bigger than I estimated (TORQUE rows skew(Bb)R^T + EXPLORATION
inflation: sampled dlam~cap misaligned) -> leak2~38/step vs manifold d2~1.7 -> 22x. FIX: leak_w 1.0 -> 0.05
(~20x down, not a halving). team_ep_R back to -40k..-85k, critic_loss <0.5. If entropy later collapses (policy
shrinks exploration to dodge leak, the jerk-term cheat) -> add a leak_cap (clip leak2/step like jerk_cap). Not yet.
** CRITICAL NORMALIZATION BUG (found + fixed, confirmed with data) ** — this was "it's not learning". With the
DESIRED-STATE obs added, estimate_norm rolled ONLY the default LINE, where 15/18 desired-state dims are EXACTLY
constant (the reference is EXACT -> zero noise -> std~1e-6; the SENSED load escaped this via sensing noise). On a
quintic those dims (y,z,roll,pitch,omega) reach 0.1-1.4 -> normalized to ~1.4 MILLION -> net fed garbage on every
quintic. SYMPTOM (48-iter run): LINE learned fine (1.26->0.62, matches the near-zero norm) but QUINTIC THRASHED
(0.4 iter12 <-> 1.45, never converged); mean dominated by the thrash -> "not learning". FIX (mappo.estimate_norm):
now rolls BASE (ZERO action -> episodes survive the FULL trajectory so the MOVED reference t~20-30s is covered;
random actions blow up early -> only cover t~0 -> wouldn't fix it) across n_traj=6 SAMPLED trajs -> desired-state
std 0.0035-5.0, worst normalized ~400 (was 1.4e6). Signature: estimate_norm(env,rng,pairs,n_traj=6); main call
passes `pairs`; only runs when WARMSTART=None. EVAL NEEDS NO CHANGE — eval_policy CONSUMES om/os_ (doesn't compute
it), so the fix propagates to eval automatically. MUST RESTART COLD (om/os_ baked once at startup). => the desired-
state+leak idea has NOT actually been tested yet (was broken by norm); restart is the real first test. Optional
robustness if still spiky: clip normalized obs to +-10 in collect/eval (standard PPO hygiene) — held unless needed.
CURRENT RUN CONFIG (restart after norm fix): obs 62-D (desired-state), leak_w=0.05, net (256,128,64) FUNNEL
(mappo HIDDEN, actor+critic), WARMSTART=None cold, cap_lam=0.5, overspeed_w=0.2/v=4, stall regime-1 (50), 35k
steps, GUARANTEE_ANCHOR=2. WATCH: does quintic DET loop now DESCEND (not thrash) by ~iter15-20; then does coord
STAY LOW (~0.05-0.15) as loop drops toward 0.33 (=coordination win) vs climb to gt1's 0.35 (leak_w too weak) vs
loop plateau high w/ low coord (frontier: shared-ref insufficient). Compare diagnose_f2 (base_clean 0.013 /
base_noisy 1.10 loop,0.036 load,0.047 coord / gt1 residual 0.33 loop,0.086 load,0.35 coord).
STALL: R2 (stall_w 50->300) planned AFTER this run, WARM on whichever base has best diagnose_f2 loop/coord (picking
gt2 = revert to 44-D/no-leak arch; picking new net = carry it). R2 often DEGRADES loop but stall is a HARD fixed-
wing constraint (below-stall = crash) not tradeable; the loop it "costs" was partly UNPHYSICAL (near-stall lingering
a real plane can't do) -> post-R2 loop is the honest flyable number. R3 margin (~0.05) = safety buffer vs grazing.

INFRA / SCALING (discussed 2026-08-23, user will circle back — NOT built yet):
* BOTTLENECK = the FMU SIM (sequential CPU physics + BLAS/torch thread-thrash on tiny matrices). ~200s/iter,
  ~35k sequential plant.step. Rollout dominates; the PPO update is a small slice.
* GPU = NO. FMU is compiled CPU physics (untouchable by GPU); nets are tiny (~58k) -> per-step transfer overhead
  >> compute (memory already: "cpu beats gpu, transfer was the 37->13s cost").
* MACHINE = 6 PHYSICAL cores / 12 logical (HT ~+20-30% not 2x). Single process ALREADY spreads across cores via
  BLAS/torch multithreading but INEFFICIENTLY (3 cores ~100%, rest >50% = overhead/contention on tiny ops).
  python.exe ~400MB/proc (FMU incl) -> MEMORY NOT a blocker (the 80% RAM is bg apps: VSCode/browser).
* LOCAL MP = single-threaded workers (OMP_NUM_THREADS=1 + torch.set_num_threads(1)) x ~5-6 (MUST single-thread or
  N-workers-x-all-core-BLAS OVERSUBSCRIBES = slower). Realistic ~2-3x ONLY (cores already busy, 6-core wall), not
  4-6x. Modest interim win.
* HPC (SLURM, 32-128 cores/node) = the REAL scaling: 10-30x + parallel seed/sweeps. The 6-core laptop is a hard wall.
* PLAN: build PERSISTENT single-threaded WORKER POOL for collect (broadcast actor state_dict/iter, aggregate per-
  worker buffers, per-worker seed=base+rank) -> VALIDATE matches single-process on a short run -> LIFT SAME CODE to
  SLURM (--workers=$SLURM_CPUS_PER_TASK + checkpoint/resume for walltime). The local step is a CORRECTNESS HARNESS
  for the cluster + a 2-3x bonus. env already MP-safe (traj = instance attr). Do local first, don't debug MP+SLURM
  at once. Offered the collect refactor; user parked it.
* AXIS 2 (fundamental, follow-up only): replace FMU with a VECTORIZED ANALYTIC plant (numpy->JAX/torch, GPU-batched
  -> 100-1000x, Brax/MJX style). Most machinery exists (LocalModelAgent reconstructs G/N/EOM). Does NOT break F2
  (desync != model-mismatch). COST = validating it matches the FMU (the FMU becomes the ORACLE). Multi-week; only if
  science justifies. FMU-first was NOT a mistake: right for the correctness-critical early phases (IL/F1 all doable,
  bet paid off = faster results); now the validation oracle.
* ADVANTAGE FILTERING (paper: drop 50% lowest-|A|) — REJECTED here: on-policy + rollout-dominated (update isn't the
  bottleneck -> no wall-clock gain; discards freshly-collected data); low-|A| already ~0 gradient in PPO; risks
  biasing adv-norm + starving critic + over-weighting blowups(-100=huge|A|)/hard-traj (undoes the critic-target fix).
  If ever: actor-only, exclude blowups, adv-norm on full batch. It's an OFF-policy/replay idea.
* CURRICULUM as separate warm runs vs one annealed run: both valid. Separate runs (current) = right for RESEARCH
  (inspect/tune/attribute each stage) but pay critic-shock/Adam/log_std/best_reward reset + config-drift each boundary.
  One annealed run (ramp weights over iters) = the "ideal" ONCE the schedule is known (continuous critic/optimizer,
  no shocks). Separate runs discover the schedule; annealed run executes a trusted one.

===== DELTA-HISTORY OBS + FUNCTION-PRESERVING WIDEN (2026-08-25) — POMDP memory, warm-start kept =====
CONTEXT: regime-2 warm-start (gr2/gt2) is STUCK — manifold_w=0, stall_w=400, coord_w=1 and it still can't
leave the basin (DET_R drifts -0.53->-0.62, loop RISES 0.27->0.34 over 40 iters). Confirms it's NOT reward
spec — it's a memoryless net in a POMDP (+ the flat warm-start gradient). USER DIRECTION: (a) drop the
prescribed-circle loiter (my anchor-orbit) for a HEADING-ROTATION reward ("velocity vector must keep sweeping,
never repeats recent direction, while coordinated" — purely LOCAL own-velocity-history, needs ZERO agreement
to satisfy so it can't be corrupted by desync; coordination stays a SEPARATE term) — NOT YET IMPLEMENTED,
see [[f2-loiter-reward]]. (b) ADD HISTORY to the obs — the change we did FIRST (one change at a time).
DONE (2026-08-25): residual obs 44 -> 80. Append TWO delta frames (18 each) of the (own6 + load18) 24-subset:
  frame layout(18) = d_load_pos(3), so3_log(dR)(3), d_load_vel(3), d_load_w(3), d_own_pos(3), d_own_vel(3).
  Frame1 = cur-prev, Frame2 = prev-prev2. Computed IN residual_marl_env._build_obs (per-drone, from each drone's
  OWN sensed/delayed stream); rolled via self._hist1/_hist2 (init None in reset -> zeros at reset/first steps).
KEY DETAIL — ROTATION DELTA: R is a matrix; elementwise ΔR is meaningless. Use the RELATIVE rotation logged to
  a 3-vec: so3_log(R_prev^T @ R_cur) (added as static _so3_log, small-angle safe) ≈ ω·dt. So R's 9 numbers
  collapse to 3 in the delta (past frame 18 not 24). Force/clock carry NO history. DELTAS chosen over raw
  stacked frames: better normalized (small, zero-mean) + strips redundancy; LOSSLESS (cur+deltas reconstruct
  the past absolutes = full delayed sequence). This is history-IN-THE-POLICY (learn a robust output-consistent
  mapping), the OPPOSITE lever from the failed KF-reconstruct-shared-truth [[f2-estimation-dead-end]].
CAPACITY: 128 hidden into 80 is fine (no rule hidden>input; stacked history is highly redundant, intrinsic dim
  ~30-40; same policy complexity, just richer input). Widen INPUT only first (observability fix, not capacity);
  widen hidden LATER as a separate experiment only if it plateaus.
WIDEN (function-preserving, warm-start KEPT): `widen_checkpoint.py` — zero-pad the actor's first Linear
  body.0.weight from [128,44]->[128,80] (append 36 ZERO columns), extend obs_mean/obs_std by 36 (mean0/std1).
  New inputs contribute NOTHING at init -> widened net = gt2 EXACTLY; the 36 delta dims get recruited by
  gradient. Critic UNTOUCHED (its input is the true global state, history doesn't change it). Produced
  residual_mappo_gt2_wide.pt. VERIFIED IDENTICAL both ways: (1) net-level max|Δout|=0 over random obs w/
  arbitrary history dims; (2) system-level real 3-step rollout, history block nonzero, gt2-on-first-44 vs
  wide-on-full-80 = 0.0 action diff. WARMSTART NOT changed yet (user: "prep, verify, save" only) — but the
  env now emits 80-D so the run's WARMSTART must point at gt2_wide (or re-widen the intended 44-D base) or
  load_state_dict crashes. NEXT: user sets WARMSTART, runs regime-2 with history; then the heading-rotation reward.

===== DIAGNOSIS PIVOT: warm-start DECAY + zero-action DECOMPOSITION + SHARED-REFERENCE anchor (2026-08-25) =====
HISTORY-OBS RUN (warm gt2_wide, regime-1 manifold=1/stall=50-hinge/load=10, no leak): FAILED like the reward-only
runs — iter1 (=gt2, history-inert) is BEST, then DET loop + DET_R DECAY monotonically (long-q loop 0.34->0.52).
Not a flat local min — it's warm-start DECAY. Key tell: DET degrades while sampled team_ep_R is FLAT NOISE (user
corrected my "sampled improves" overclaim) -> no improvement signal anywhere; adv-normalization manufactures unit-
variance from ~zero-signal noise -> random-walk erodes the good warm-start.
WHY NO LEVER MOVES IT (the pedagogy, delivered): 3 requirements for RL to improve = reachability + observability +
signal. More data attacks VARIANCE (can't create signal); reward reshape re-weights terms the policy can't move
favorably (incentive w/o ability); history adds observability but of a channel that's ~SILENT on the missing
quantity. All 3 fail identically = INFORMATION bottleneck, not optimization -> looks like a stubborn min but is the
CONSTRAINED optimum for the info a decentralized agent has. Coordination was free clean (identical view->agreeing
lambda); desync breaks the shared input -> can't reconstruct agreement from divergent PRIVATE views.
USER CAUGHT TWO REAL ERRORS (I conceded): (1) "load is blind to the leak" is WRONG — disagreement makes the
assembled internal force a PATCHWORK that is NOT a clean nullspace vector -> it HAS a range component -> DOES
wrench the load (leak=||G@f_int|| measures exactly this; load 0.036->0.086). Only the CONSISTENT nullspace DRIFT
(dominates loop) is load-invisible; the LEAK is load-VISIBLE but REGULATED (PID rejects most -> low SNR, muffled
NOT blind). (2) "flat terrain / no hill" too strong — it's a loop<->coord PARETO frontier; residual->0 recovers
base coord (reachable), so not flat. History's estimate->agreement path IS real; its gain is bounded by the
DELAY-divergence (each drone's view delayed by unknown time-varying d_i; can't time-align w/o the delay = the
KF-dead-end wall) + attenuation + drift confound -> small, buried, not zero.
ZERO-ACTION DECOMPOSITION (zero_probe.py, scratchpad; NO training; under manifold=0/stall=50/load=10/leak=0.1):
decisive tool. Per-TERM weighted per-step penalty, base vs policy, on line + held-out quintic. RESULT — it's a
WASH on the eval mean (base -0.7175 vs policy -0.7185 = -0.18/agent = the run's iter1 DET_R). My "walk to zero"
bet's PREMISE was wrong: base STALLS+OVERSPEEDS on the line so the residual has a real job even w/ manifold off.
THE MECHANISM (complementary terms): base's penalty = STALL + OVERSPEED (desync SCATTER: some drones too slow,
some SPIKE >4m/s); policy's penalty = LOAD + LEAK (it STABILIZES the scatter but disturbs the regulated load
doing so). Residual TRADES drone-scatter for load-disturbance. Line: base is a mess (stall 0.25 + overspeed 0.43)
-> residual's stabilization worth its load(+0.15)+leak(+0.22) cost -> policy WINS. Quintic: base already clean
(stall/overspeed ~0) -> residual only ADDS load(+0.25 DOMINANT)+leak(+0.06), nothing to fix -> policy LOSES.
THREE TAKEAWAYS: (a) manifold=0 HIDES the residual's main quintic value — it DOES fix loop 1.1->0.32, but that
lives in the 0-weighted manifold term, so reward sees only the load side-effect -> this config structurally
misjudges the residual = dead end. (b) leak_w=0.1 is a rounding error; the residual's real coordination tax is the
LOAD term (load_w=10 already captures ~80% of it). (c) NEW: base OVERSPEEDS (spikes >4m/s from scatter) and taming
it is the residual's BIGGEST single win on the line (0.43 > the 0.25 stall saving) — hadn't isolated that.
NOT A TRILEMMA (user pushed, correctly): base_CLEAN does all 3 (no stall/spike, load stable, loop 0.013) = EXISTENCE
PROOF it's possible. Stall/spike + load-disturbance are TWO FACES of ONE deficit = drones disagreeing on the
nullspace config; fix agreement -> all improve together. Current residual HALF-coordinates (stabilizes locally, but
misaligned corrections land on load). Open (NOT proven impossible) = how much agreement a zero-comms policy can
manufacture from private noisy/delayed views; easy routes (KF, own-history) don't, but gt2's ~0.3 isn't proven the
ceiling. The RIGHT lever = attack AGREEMENT directly: OUTPUT-CONSISTENCY (train noisy-view lambda ≈ clean-view
lambda, privileged-at-train) [most promising, untried]; SHARED-REFERENCE anchoring; or inter-drone info.
SHARED-REFERENCE ANCHOR ADDED (2026-08-25, user chose full 18-dim): obs 80 -> 98. Appended [p_d,R_d(9),v_d,omega_d]
= desired load state from get_reference_trajectory at each drone's mission clock, laid out like load18. It's the
ONE signal IDENTICAL across drones (all share t; clock_offset=0) -> anchors coordination. ADDED not substituted
(pure-ref goes open-loop, refbase probe) so the policy blends anchor + noisy-view feedback. CRITICAL norm handling:
ref dims are LARGE/varying (p_d to 11m, CONSTANT on the line) -> CANNOT use the history's mean0/std1 zero-pad or it
reproduces the old desired-state norm bug; widen_checkpoint computes REAL mean/std ANALYTICALLY from
get_reference_trajectory over all 56 trajs (it's a deterministic fn of (t,traj), no rollout) -> std [0.004,4.02].
widen: gt2(44)->98, body.0.weight zero-pad 54 new cols (36 hist mean0/std1 + 18 ref real), critic untouched.
VERIFIED: net-level max|dout|=0; ref block == get_reference_trajectory (allclose); system-level gt2(44) vs
wide(98) real-rollout action diff 0.0. residual_mappo_gt2_wide.pt is now 98-D. WARMSTART still not set by me;
env emits 98-D so the run MUST warm from gt2_wide. This was ATTEMPTED once before (the +18 desired-state run) but
never cleanly tested (norm bug then reverted) -> this is the FIRST clean shot, on the warm base, norm done right.

RESIDUAL TRANSFERS (user corrected me 2026-08-18):

RESIDUAL TRANSFERS (user corrected me 2026-08-18): the residual does NOT see the F1 net's 66-D input -- its
obs is the 44-D [load18, own6, f_g3, f_lam3, clock14], dims UNCHANGED, so residual_mappo*.pt loads fine. Of
that obs only `f_lam` (nullspace slice, from the F1 lambda) shifts; `f_g`(=G+w_d, PID) + state + clock are
identical, and base tracking is unchanged. The recent RL nets are CLOCK-ANCHORED (keyed to phase, the coord
signal), so they should just work on the new base. No forced retrain -- only a quick eval to confirm the small
f_lam / drone-speed shift didn't move anything at the margins (stall).
