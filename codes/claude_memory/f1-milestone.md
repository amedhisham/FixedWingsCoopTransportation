---
name: f1-milestone
description: "Formulation-1 decentralized lambda policy WORKS — IL + clock + DAgger flies clean, replaces the optimizer"
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-08-10T10:48:18.806Z
---

As of 2026-07-19, **the F1 METHOD is proven as a proof-of-concept on ONE trajectory**
(NOT generalizing yet): a single shared policy outputs the internal-force coefficient
λ per drone (replacing the CasADi optimizer) and flies clean in closed loop. Final
`deploy_f1` on `il_actor_dagger.pt`: load tracking mean 0.081 m, drone velocity norms
~1.40–1.43 m/s (well above ε=0.25), symmetric across the φ=[0,π/2,0,π/2] pairing
(drones 1&4 identical, 2&3 identical). Velocity plot smooth — the closed-loop "buzz"
is gone.

CAVEAT (user-flagged): trained/evaluated on the SINGLE fixed reference (static → +x
move → static, fixed ICs). The policy has likely MEMORIZED this scenario, not learned
a general loiter law. It is de-risking, not deployable. The immediate next work is
**trajectory + initial-condition randomization** (parameterize
`controller.get_reference_trajectory`, collect + DAgger over the distribution), BEFORE
F2. Open watch-item: the clock is `sin(ωt)` in ABSOLUTE time — phase↔time alignment
may not survive randomized timings/durations and may need rethinking.

**Speed result:** deploy_f1 sim loop (policy replaces optimizer, CasADi-free) ~1.54 s
vs main.py centralized loop (CasADi optimizer) ~16 s — both loop-only, 2500 steps, same
machine → ~10x. BUT the fair DECENTRALIZED comparison is bigger: main.py does ONE
central solve/step, whereas a real decentralized classical system needs 4 INDEPENDENT
CasADi solves/step (each agent, own view — can't share IPOPT). deploy_f1 already does 4
inferences/step, BATCHED (4,38)->(4,1) ≈ cost of one. So 4 batched inferences (~1x) vs
4 serial solves (~4x) → ~40x, and the gap SCALES with fleet size (N inferences batch to
O(1); N solves are O(N)). Exact ratios are a TODO (time 4 independent solves; caveats:
main.py ~11 per-step getReal vs wrapper's one batched read, epsilon 0.20 vs 0.25).
Defensible headline: "tens of times faster, and growing with N."

The pipeline that got there (order matters):
1. **IL warm-start** (`train_il.py`) — behavior-clone λ from the adaptive optimizer.
2. **Clock Fourier bank in the obs** (`collect_il_data.build_obs_rows`, 38-D obs =
   18 load + 6 own drone + 14 clock `[sin,cos]` over ω∈linspace(1.5,3,7)). The clock
   is the DAgger *enabler*: it makes off-orbit labels a consistent function of the
   obs (kills the "same position, different time → different λ" collision), so
   aggregation converges instead of flooring at Var(λ).
3. **DAgger** (`dagger.py`) — β-mixed rollouts (β 0.9→0), expert-labeled, aggregate +
   retrain from scratch each iter. β-mixing is safe because G·N=0 (λ only touches the
   nullspace, load tracking untouched for any λ).

Key framings that resolved the design: the failure was **covariate shift + an
unsupervised nullspace loop**, not overfitting. Phase is a **carried state**, not a
readable input; the clock supplies the fast phase (drift-free), position only supplies
the slow slot identity. The policy is **feedback regulation on local ‖v‖**, not
imitation of the sweeping optimizer.

**MAJOR PIVOT (2026-07-20) — the WHOLE-VECTOR net supersedes the per-drone F1 net.**
The per-drone net (obs = load+own-drone+clock → own λ) was the SOURCE of every problem:
it keyed λ on the drifting DRONE state → closed-loop covariate shift → buzz → needed
DAgger; and for F2 it forced neighbor RECONSTRUCTION, which was UNSTABLE (sanity test
diverged ~1s into the move phase; isolation test proved reconstruction was the sole
culprit — feeding the net ~cm-off neighbor states destabilizes it, either pos or vel
channel alone). Fix (user's idea): copy the OPTIMIZER's actual I/O — a net
`(load 18 + clock 14) → λ-vector[n]`, NO carrier state. Files: collect_vec_data.py,
train_vec.py, deploy_vec.py, il_actor_vec.pt. Results: BC fit R²=1.000 (vs 0.994 for the
per-drone net — MORE info fit WORSE), and deploy_vec flies closed-loop IDENTICAL to
deploy_f1 (mean 0.0813, max 0.3868, vel ~1.40-1.43) with NO DAgger and NO reconstruction.
Why: the load is PINNED by the wrench controller (G·N=0 → tracks for any λ), so the net's
input distribution is stable → no covariate shift → no buzz. This is why the optimizer was
always stable (fed on load, not drones); the network inherits it. Phase comes from the
clock, the per-drone φ are baked into the output heads (no slot inference from position).
For F2: each drone runs this net on its OWN load estimate + clock → λ-vector → uses λ_i and
λ_{i-1}; desync = load-estimate/clock divergence. NO reconstruction anywhere.
DONE (2026-07-20): residual_marl_env rewired — LocalModelAgent is now just wrench +
vec_net(own load estimate + clock) → λ-vector → slice (all reconstruction ripped out);
loads il_actor_vec.pt, act_dim=n. sanity_f2.py at zero noise/zero residual FLIES clean
(0.0814/0.3868, vel ~1.40-1.43) = matches deploy_vec, no blow-up. F2 base is stable+correct.
Next: turn on desync knobs (pos_noise/clock_offset/ctrl_delay) to see degradation, then build
the MAPPO residual (δf) trainer.

**F2 architecture (confirmed by user):** F2 is a SEPARATE residual network adding δf on
top, using the (frozen) F1 policy as the BASE *instead of the optimizer* — NOT a retrain
of F1. Per agent: w_d (analytic) + λ from F1 net + δf from F2 residual net → total force,
trained by RL (CTDE/MAPPO) for desync robustness. Crucially δf is a FULL force (not
nullspace-restricted), so it HAS the authority to counter observation-desync load
disturbance — which λ alone (nullspace-bound) does not. NO CasADi anywhere in F2 (the
whole reason F1 came first: RL needs thousands of rollouts; CasADi-in-RL is infeasible).
Desync knobs already exist in residual_marl_env.py (pos_noise, clock_offset, ctrl_delay,
_update_estimates). Real danger is observation desync → inconsistent G_i/N_i/w_d_i →
forces don't cancel → load disturbed; clock desync mostly sits in the nullspace (benign).

**Zero-communication decentralization (confirmed viable, 2026-07-19):** each drone needs
its own λ_i AND its ring-predecessor λ_{i-1} — the nullspace N is proven SPARSE (each
drone's 3 force rows depend on exactly 2 λ's: own + cycle-neighbor; verified numerically,
N=(H⊗I3)@D with H the incidence matrix, 2 nonzeros/row). Zero-COUPLING is impossible
(G·N=0 forces internal forces to coordinate). But zero-COMMUNICATION is achievable: each
drone RECONSTRUCTS every carrier's pos/vel from its own load estimate + geometry + the
(filtered) forces via the paper's kinematics — p_R=p_L+R·Bb_i+L0·q_i, ṗ_R=v_Li+(L0/T)Π ḟ_i,
q=f/‖f‖, q̇=Π ḟ/T — then queries the frozen F1 net locally. No neighbor observation, no
explicit phase, NO RETRAIN (net just gets computed carrier states instead of observed).
verify_reconstruction.py CONFIRMS it: steady-state error ~1.3 cm pos / ~3 cm/s vel vs the
plant (max 0.8m=L0 is a t=0 startup transient only). Small residual → the F2 δf absorbs it.

Plan: do F2 on THIS single trajectory first = full-pipeline PoC (retire the research risk
cheaply before the trajectory-randomization grind). Optional RL fine-tune of F1 (local
‖v‖≥ε + smoothness) is separate polish. See [[project-overview]].

**pR_dot ARCHITECTURE (2026-07-20) — the generalization-ready policy, DAgger'd clean.**
Motivation (user): the load+clock whole-vector net MEMORIZES the schedule on one
trajectory (R²=1.0 is a lookup, not a law); it can't know when to loiter harder from load
state alone. The optimizer's real decision variable is the carrier velocity v_Ri (Eq. 22) —
so feed the net THAT. New I/O: `[clock(14), pR_dot(n*3), lambda_{t-1}(n)] -> lambda[n]`.
Roles: clock=phase, pR_dot=amplitude feedback (the constraint variable), lambda_{t-1}=
smoothness/anchor (pR_dot is lossy about it). pR_dot is RECONSTRUCTED analytically EXACTLY
as the optimizer computes v_Ri (f=G+w_d+N·lam_{t-1}, fdot=e_total+Ndot·lam_{t-1}+N·lamdot_{t-1},
tension floor sqrt(‖f‖²+1e-6)) — NOT from the LLC-filtered plant force, and NOT from
observed drone velocities (the net can't see the other 3 drones at runtime — this is the
whole reason to reconstruct; it's the only deploy-viable, decentralization-compatible input).
lamdot via finite-diff of the lambda history so collect/deploy compute it identically.
Files: collect_prdot_data.py (shared reconstruct/build_input helpers), train_prdot.py,
deploy_prdot.py, dagger_prdot.py. BC fit R²=1.000 but deploy BUZZED (predicted): the net
self-feeds pR_dot from its OWN lambda_{t-1}, which drifts off the optimizer-lambda manifold
the dataset was built on = covariate shift. FIX = DAgger (dagger_prdot.py): roll out the
policy but reconstruct its pR_dot input from the ACTUALLY-APPLIED (beta-mixed) lambda history
(the state it truly visits), label with the optimizer, aggregate, retrain; beta 0.9->0 slides
the training distribution onto the deployment (self-fed) one. beta-mixing safe (G·N=0).
RESULT (2026-07-20, user-confirmed): "we cooked, the plots are perfect" — DAgger'd pR_dot
policy (il_actor_prdot_dagger.pt) flies clean closed-loop. This is the architecture to carry
into trajectory/IC randomization (pR_dot is causal + load-derivable, unlike the clock-memorized
load+clock net). Next real test: multi-trajectory generalization.

**F2 DECENTRALIZED STRUCTURE VALIDATED (2026-07-20).** residual_marl_env rewired to the
pR_dot policy: each drone runs its OWN stateful LocalModelAgent (reconstruct all N pR_dot
from its own load estimate + own lambda history -> net -> keep its own FORCE slice; force
calc runs N times, once per drone, each applies its own slice — NOT a lambda slice). Split
into prepare()/finalize() so the N per-drone input rows run in ONE batched net forward, and
finalize reuses the G+/N from reconstruct instead of recomputing (dropped the redundant
cable_force_calculation). sanity_f2 at zero noise/zero residual flies clean (matches
deploy_prdot) with the 4 deploy-style plots. Benchmark: ~2.4 ms/step, 4.1x real-time (was
4.6 ms/step / 2.2x before batching+G+N-reuse). Cost is overhead-bound (torch dispatch,
pinv/nullspace, FMU doStep), NOT MLP FLOPs. Next: turn on desync knobs (pos_noise/
clock_offset/ctrl_delay) to see load disturbance, then the MAPPO residual (delta_f) trainer.

**DESYNC ROBUSTNESS OF THE BASE (2026-07-21..22).** Under observation desync the base's
velocity went to trash. Chain of findings + fixes:
1. **Jitter localized to the RECONSTRUCTION** (jitter_diag.py): the analytic `reconstruct`
   finite-differences noisy w_d/lambda (÷dt = ~100x gain) -> pR_dot jitter 220x under desync.
2. **FIX = low-pass reconstruction `reconstruct_lp`**: low-pass the LOCAL base force
   f = G+ w_d + N*lambda (decentralization-legal, no neighbour delta_f) then differentiate
   the SMOOTH result -> pR_dot jitter 220x->2.7x. Filter time const `recon_tau` is a TUNABLE
   ESTIMATE of the drone LLC (decoupled from the plant's true 0.2s; we don't know it exactly).
   Chose recon_tau=0.1 (lighter = less lag, jitter still tiny). RECON_ALPHA=DT/(0.1+DT).
3. **Retrained base on filtered pR_dot** (collect->BC->DAgger): R2=1.0, flies clean at ZERO
   noise. But under desync it JITTERS AGAIN — the sharp DAgger'd policy re-excites the lambda
   self-feedback loop (lambda_{t-1}->pR_dot->lambda_t); pR_dot is NOT policy-independent
   (built from N*lambda_{t-1}), loop gain ~1 -> limit cycle. il_actor_prdot_dagger.pt.
4. **Noisy DAgger** (dagger_noisy.py): decentralized rollout WITH desync, per-drone noisy
   views (DesyncSensor), label = CLEAN optimizer lambda on TRUE state -> policy learns to
   denoise. Reduced buzz (esp early episode) BUT velocity still bad late; policy matched
   lambda "okay-ish" (MSE 0.0004) yet velocity trashed -> the MSE-blind-to-jitter ceiling
   (see [[f2-residual-rl-plan]]). il_actor_prdot_noisy.pt. => base imitation is capped; RL next.
5. **ANALYTIC flag + Reconstructor** (collect_prdot_data): toggle in collect/train/deploy/
   dagger between low-pass (default, SUFFIX="") and original finite-diff (`ANALYTIC=True`,
   SUFFIX="_analytic"). Reconstructor class encapsulates each mode's per-step state so the
   loops are mode-agnostic (`vR=recon(...)`, `recon.roll(applied_lam)`). ResidualMARLEnv does
   NOT use the flag — LocalModelAgent calls reconstruct_lp DIRECTLY (env is ALWAYS low-pass).
   Recreated the OLD analytic policy under _analytic names (il_actor_prdot_dagger_analytic.pt)
   to compare, as a residual base, SMOOTH-BUT-DRIFTING (analytic policy run in filtered env =
   off-distribution = insensitive = smooth) vs the filtered-trained-but-JITTERY one. A smooth
   base is likely the better residual base (residual fixes drift easily, jitter hard).
Files added this arc: jitter_diag.py, dagger_noisy.py, recreate-via-flag. residual_marl_env
gained recon_tau param + infos["prdot_own"]. Next: pick the base, then F2 [[f2-residual-rl-plan]].

===== F1 GENERALIZATION — MULTI-TRAJECTORY (2026-08-03..10) =====
**The whole F1 pipeline (collect/train/dagger/deploy) was GENERALIZED from ONE trajectory to a
DISTRIBUTION of paper-Fig.10 quintic 6D pose trajectories.** See [[f2-residual-rl-plan]] GENERALIZATION
PHASE for the trajectory decision + base-readiness evidence (examine_base.py: base tracks gentle quintics
to ~1-2cm, no accel-FF needed; harsh arcs were the artifact).
**FILES:** trajectories.py (NEW single source of truth: sample_traj/train_set(K,seed)/heldout_set(M) ->
rest-to-rest QUINTIC 6D pose via controller.make_quintic_pose; knobs POS_RANGE=1.5m, ROT_RANGE_DEG=10,
RAMP=10, HOLD=5, MIN_POS_NORM=0.3; TRAIN_SEED vs disjoint HELDOUT_SEED). controller.py: get_reference_
trajectory(t, traj=None) delegates to traj; make_quintic_pose(pos_delta, rot_delta, ramp, hold, base_pos,
base_R) w/ ROTATION (scipy R_tool.from_rotvec, omega=delta_r*sd body-frame, correct for fixed-axis).
TRAJECTORY IS ENV-THREADED not a global: error_calculation(...,traj)/compute_forces(...,traj)/env.traj
instance attr -> parallel-safe (SLURM/multiproc ready). collect_prdot_data: rollout(env,agent,Bb,L0,traj)
+ collect(n_traj=N_TRAJ=50, include_default=True), reuses ONE env+agent (agent.reset between eps).
dagger_prdot: rollout(...,env,agent,Bb,L0,traj) reused; TRAJ_PER_ITER=10 + INCLUDE_DEFAULT anchor per iter;
RESUME-AWARE (loads il_actor_prdot_dagger{SUFFIX}.pt + AGG_OUT=prdot_dagger_aggregate{SUFFIX}.npz if they
exist, else BC+collect) and SAVES the aggregate -> real continue; BETAS=[0.0]*K for pure-policy continue.
train_prdot: UNCHANGED (data-driven). deploy_prdot: run_episode/plot_episode/main(traj)/evaluate_heldout(M);
POLICY=f"il_actor_prdot_dagger{SUFFIX}.pt", EVAL_HELDOUT=True.
**KEY FINDINGS:**
- F1's lambda-map is TRAJECTORY-AGNOSTIC (local state -> optimizer lambda), so open-loop MSE stays ~0
  (R2~1) on held-out; BC converged in ~4 EPOCHS (not warm start - fresh init; 50x more samples = 50x more
  gradient steps/epoch, "epochs to converge" misleads when dataset sizes differ 50x).
- GENERALIZATION WORKS: HELD-OUT quintics track at CM level (0.004-0.029 m), no train/held-out gap in
  either open- OR closed-loop MSE (held-out ~= train, 0.8-0.9x).
- The "held-out garbage" the user saw was TWO RED HERRINGS: (1) a STALE-CHECKPOINT BUG - deploy_prdot
  defaulted to hardcoded "il_actor_prdot.pt" = a JULY non-analytic net, NOT the trained _analytic one
  (FIXED: POLICY now SUFFIX-aware). (2) the auto-deploy runs on the DEFAULT straight-line, which is the
  HARDEST trajectory in the mix (v=1.1 m/s + velocity STEPS, harsher than the gentle quintics) -> worst
  track (0.10) + buzz + vmin dips to 0.111 (<eps 0.25). Held-out gentle quintics are the EASY ones.
**CURRENT OPEN ISSUE (2026-08-10): closed-loop BUZZ plateau.** ~10 beta=0 DAgger iters IMPROVED buzz but
STAGNATED (still improving iter 8-9, then flat). Residual buzz is TIME-LOCALIZED: first ~10s good, LAST 5s
(STATIC-HOLD phase t=15-25, load parked, drones must loiter) still buzzes. Same on train+held-out -> NOT
a generalization gap; it's the self-fed pR_dot off-manifold buzz. Coverage (more DAgger) = diminishing
returns -> points to CAPACITY/MEMORY: a memoryless MLP (only 1-step memory via lambda_{t-1}) can't DAMP
its own closed-loop oscillation, worst in the static-loiter phase. Leading next lever = MEMORY (GRU/stacked
history) > width (buzz is a temporal/stability problem). Ties to the [[f2-residual-rl-plan]] bottleneck-
diagnosis thread (memory-vs-capacity, arch sweep in RL). ASIDE (user chasing separately): a v_Ri mismatch
between the OPTIMIZER (analytic Eq.22, L0/T rigid-cable) and the FMU (elastic cable + it computes carrier
vel from the force-DERIVATIVE input) - user said forget for now, will look into it.
