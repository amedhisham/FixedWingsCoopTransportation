---
name: f1-phase-slip-lambda-history
description: F1 fails by a SUDDEN phase slip into anti-phase lock (not gradual drift); root cause is a memoryless net imitating a stateful A/xi-ratcheting expert. RESOLVED 2026-08-18 by LOG-SPACED lambda-history taps (straddle the flip) + the save-bug fix; F1 works closed-loop, moving to F2.
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-08-18T14:57:55.032Z
---

The current F1 failure mode + the fix shipped 2026-08-16. Continues [[f1-dagger-warmstart-gate]],
[[f1-antimemorization-plan]].

## The failure (from documentation/lambdaNetvsExpert_f1a2_fail.png)
The net tracks the optimizer's lambda sinusoid WELL (correct phase AND frequency) for a while, then a
buzz episode erupts, then a SUDDEN ~half-cycle PHASE SLIP develops and it LOCKS anti-phase, never
recovering. NOT a gradual drift (I was wrong about that first). Slip timing is VARIABLE and not tied to
any event: short_line slipped ~15s, short_quintic1 ~20s, both with move-end at 15 — so it's NOT
transition-triggered and NOT a fixed clock time; the buzz just accumulates until it kicks a slip.

## Mechanism (corrected, high confidence)
Frequency is fine (the clock anchors xi); it's the PHASE that slips. A slip is only POSSIBLE because
the net sets phase from its OWN lambda feedback, which admits MULTIPLE stable locks — including
anti-phase, via an approximate lambda -> -lambda symmetry in the pR_dot reconstruction (pR_dot magnitude
depends on lambda's rate/size, ~not its sign). The clock is monotonic absolute time and could NOT slip;
the feedback can. Buzz kicks it across the barrier. beta=0.04 is a BIFURCATION not a special number:
above it, `beta*expert` re-injects the correct phase each step and heals an incipient slip; below a
critical beta the anti-phase attractor captures it (2-orders-of-magnitude blowup). Same slip seen at
0.1/0.08 sometimes -> same bifurcation, different threshold.

## Root cause = POMDP: memoryless net vs STATEFUL expert
The optimizer carries hidden integrator state (A, xi) that ratchets UP and PERSISTS (deadband: rare
engagements). A/xi are only WEAKLY observable from `[clock, pR_dot, lambda_{t-1}]`: lambda_{t-1} encodes
theta=xi*t but wrapped mod 2*pi, so xi isn't robustly recoverable over a long horizon. So the same
visible input maps to different lambda depending on unobserved accumulated xi -> the target isn't a
function of the inputs. Worse, the adaptive (ratchet) events are <1% of the data (SPARSE) — user's main
worry. NB the net's WHOLE job is that adaptation (raise A when pR_dot nears eps to keep drones moving);
the optimizer proves the job exists. MSE<->buzz strongly-but-not-100% correlated (see [[f1-dagger-warmstart-gate]]).

## FIX SHIPPED + RESOLVED (2026-08-18): lambda-HISTORY, LOG-SPACED taps
F1 now WORKS closed-loop at beta=0, deploy-verified (user 2026-08-18). Net sees its last APPLIED lambdas
(newest-first), not just lambda_{t-1}: PHASE INERTIA (a continuous recent trajectory can't jump to
anti-phase without a discontinuity the net never saw -> resists the slip) + makes A/xi observable.
FINAL FORM = LOG-SPACED point taps, NOT dense last-10: the dense 0.1s window was ~15x shorter than the
~1.5s half-cycle flip, so it couldn't STRADDLE a slip (by the time the buffer was anti-phase the flip
already happened). Now retain a FULL-DEPTH ring buffer (LAM_MAXLAG=256) and expose 10 taps at
`LAM_LAGS=[1,2,4,8,16,32,64,128,192,256]` steps (0.01..2.56s ~= one lambda period): dense recent for
step continuity + deep taps that reach past a half-cycle so the buffer straddles the flip. Same 10-tap
budget so input stays 66-D (clock14 + pR_dot12 + 10*4) — but semantics changed so a FRESH BC train was
needed (a dense-window net misreads taps). Impl in `collect_prdot_data`: `init_lam_history(depth=256)`
= analytic continuation `A0*cos(XI0*(-k*DT)+PHASES)` (seamless: during the hold the real lambda IS that
sinusoid; LAM0 is just its t=0 sample), `push_lam` full-depth ring, `lam_taps(buf)=buf[_LAG_IDX]`,
`build_input` taps 2-D buffers / flattens 1-D (F2's residual_marl_env still passes 1-D prev_lam -> 30-D,
untouched). Shared by collect/dagger/deploy so no train/deploy mismatch. This + the save-bug fix
(see [[f1-dagger-warmstart-gate]]) is what got F1 clean closed-loop. Now MOVING TO F2 (mappo residual).
Aux-A/xi / phase-supervision / self-conditioning levers were DISCUSSED but NOT needed — taps sufficed.

## Levers for the SPARSITY (memory alone won't create ratchet examples)
1. Memory [DONE] also AMPLIFIES the sparse signal: each rare ratchet, once integrated into the persistent
   history/state, shapes the WHOLE rest of the episode instead of 1 step -> far more training signal.
2. UPWEIGHT the hard/solver-engaged samples in training (~20% of each batch) — the curate rebuild
   [[f1-curate-redesign]]. Becomes essential if we ever make A/xi an explicit target (class imbalance).
3. BOUNDARY-RIDING trajectories: keep min||v_Ri|| sitting NEAR eps for a sustained stretch -> dense
   adaptation (50%+ engagement vs the customs' ~5% brief grazes). DEFERRED (hard to hand-design).

## PARKED idea — auxiliary A/xi output (user wants lambda FREE for RL, so do NOT reconstruct lambda from A/xi)
Keep lambda as the free primary output; ADD A,xi as AUXILIARY outputs with `w*MSE(A,xi)` (labels =
`agent.prev_x`). Forces the shared representation to name a single coherent (A,xi) -> the free lambda
should come out more phase-coherent. Big value is DIAGNOSTIC: low aux MSE => A/xi ARE recoverable from
the inputs; high => they're not, so inputs must change (self-feed A/xi = a clean 2-scalar memory). It's
a training-time REGULARIZER (biases, doesn't structurally stop the slip). Order: aux head first, then
self-feed if needed. Implement LATER. (Reconstructing lambda=A*cos(xi*t+phi) with absolute t would kill
the slip structurally + drop the clock, but makes lambda a rigid sinusoid -> rejected: no RL headroom.)

## Also (from [[f1-antimemorization-plan]]): z-moves don't engage the solver (horizontal-plane phenomenon).