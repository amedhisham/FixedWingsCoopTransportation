---
name: f1-dagger-warmstart-gate
description: "Warm-start transmits a jittery net's bad weight-basin; DAgger now gates on buzz and stays at a beta until it passes. Implemented in dagger_prdot."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-08-18T14:58:20.135Z
---

Hard-won F1 DAgger lesson + the fix now in `dagger_prdot.main`. Continues [[f1-curate-redesign]],
[[f1-antimemorization-plan]].

## The revelation (user, 2026-08-14)
User accidentally warm-started a FRESH DAgger run (beta=0.9) from an OLD net that had been trained
down to ~beta=0.08 and had learned JITTER. It did MUCH worse than starting from the plain BC net,
and couldn't unlearn the trash until beta=0.6 (then gave up). **Warm-start can HURT: it transmits the
initializer's weight-BASIN.** A consolidated-jitter net is a bad (high-buzz) basin; warm-start plants
you there and you're stuck. BC happens to be a GOOD (low-buzz) basin (collection is the smooth
optimizer, no self-excitation), so warm-starting from BC is safe; from a late trash net is not.

## Why it's sticky (corrected mechanism)
NOT "MSE is blind to buzz" — user corrected me: **MSE and buzz are STRONGLY but not-100% correlated**
(buzz corrupts the self-fed pR_dot -> ill-posed target -> MSE rises WITH buzz). So training DOES have
a restoring force. The stickiness is: (1) warm-start starts you in the bad basin, (2) the trash net's
own rollouts POISON the data (even at beta=0.9 the 0.1*policy + feedback corrupts visited states), so
the MSE-min ON THAT DATA is a compromise that still buzzes — and the **non-100% slack in the MSE<->buzz
correlation is exactly where a jittery-but-lowish-MSE solution lives.** Paradox (mid-run spike at
beta=0.6 self-corrects next iter, but deep trash doesn't) = shallow excursion near a good basin
(clean-ish data restores) vs deep consolidated basin (flat, no restoring force).

## The fix (IMPLEMENTED in dagger_prdot.main)
Adaptive beta ladder: **don't leave a beta until closed-loop buzz passes**, and warm-start each retry
from the last VERIFIED-GOOD net (never a rejected candidate). Gate on the EXISTING rollout buzz metric
(`mean(buzz_k)`) — no extra eval rollout needed. Absolute threshold. Only near deployment.
- `GATE_BETA=0.1` (gate only for beta<=0.1; above -> plain one-pass-per-beta, original new-fully+
  sample-old + evict-to-MAX_AGG).
- `BUZZ_PASS=0.022` (leave a gated beta once buzz < this; floor ~0.016).
- `MAX_RETRIES=3` retrains at a stuck gated beta, then advance best-effort.
- STUCK -> KEEP ALL data (no eviction) + train on ALL of it (pool + budget grow "a bit, no problem —
  we need it"); retries draw FRESH trajectories.
- `good_state` snapshot = last accepted net; `bi/it/retries` while-loop (it = global generation for
  seed/age so retries keep the stream moving).
Structure note: the rollout at the top of each pass evaluates the PREVIOUS pass's trained candidate,
so the same rollout that generates data also gates the last candidate (no extra rollout).

## Gate refinements (2026-08-15/16)
- `STUCK_AFTER=0`: the FIRST failed gate is already "stuck" -> immediately escalate to keep-all-data +
  train-on-the-WHOLE-pool (a buzz problem needs the hard data now). Gating alone (beta<=GATE_BETA) does
  NOT bloat — a beta that PASSES stays at the normal caps (TRAIN_BUDGET=200k, MAX_AGG=250k).
- PASS-TRAINS (fixed a bug: a passing beta used to advance WITHOUT training on its fresh rollout ->
  wasted a whole rollout). Now on PASS: anchor `good_state` to the VERIFIED pre-train net, then train
  (normal budget, warm from the passing net), carry the trained net forward (next beta re-verifies it).
- `evict_pool()` collapses the pool back to MAX_AGG on EVERY advance (PASS/give-up/non-gated), so the
  stuck-episode bloat is temporary, never carried into the next beta. Shared helpers `add_new()`,
  `evict_pool()`, `budgeted_trainset()` are the single data path. Stuck retry -> train whole pool (no
  evict); non-stuck retry / advance -> budgeted_trainset (new-fully + sampled-old) then add+evict.

## RESUME/SAVE BUG — FIXED 2026-08-18
Symptom (also surfaced as "dagger per-iter plot CLEAN, deploy NOISY/cooked" on the SAME default traj):
`main()` SAVED `policy` (the last POST-train net) not the verified net. On a PASS we snapshot the
verified net then PASS-train once more with NO subsequent rollout to re-verify -> saved `policy` is an
UNVERIFIED post-train net that can be cooked. PROVEN: saved net on the default traj had applied-buzz
0.1794 (11x the 0.0161 the plotted/gated pre-train net passed at); track stayed fine (0.06, buzz lives
in the G.N=0 nullspace) so "cooked" = NOISY lambda, not blown tracking. (The `vmin=0` I first cited is a
RED HERRING — it's the t=0 initial-rest sample, ~0 for ANY net; user only cares about buzz+MSE.) Resume
also warm-started everything from that cooked net -> sticky bad basin. FIX SHIPPED: save `recent[-1]`
(the verified good_state + its MATCHING om/os_), not `policy`. Gate/pass logic UNCHANGED (still
`buzz < BUZZ_PASS`, no vmin). Secondary (why split!=full bit-for-bit, not the cooking): `curate_rng`
resets each run + `it`-reset changes the resumed quintics.

## CONTINUOUS AUTOSAVE of last-2 verified-good (2026-08-18)
So a run is NEVER lost to the above again on ANY exit path. `recent=deque(maxlen=2)`; `keep_good()` runs
wherever good_state is anchored (init, gated PASS *before* the pass-train so om/os_ still match, non-gated
advance) — appends {state_dict, obs_mean, obs_std, hidden} AND writes BOTH files every accept:
`il_actor_prdot_dagger_autosave_last{SUFFIX}.pt` (newest) + `_prev` (2nd-newest). Stuck/retry candidates
are EXCLUDED (only clean nets hit disk). Normal exit also saves `recent[-1]` as the main ckpt. Ctrl-C /
crash: files already current. So 3 good artifacts always on disk: main + _last + _prev. Also added a
`try/except KeyboardInterrupt` around the ladder (nested `try:` at 6-sp indent inside the while) that
breaks + returns. NB the collect AGGREGATE pool can still carry a bad rollout (add_new appends the
passing rollout's data); a truly clean restart = move aside the cooked dagger ckpt AND the aggregate npz
-> main() falls back to BC net (`il_actor_prdot{SUFFIX}.pt`) + clean collect data (`prdot_dataset`).

## The diag/gate plots RAW lam_pol, deploy plots APPLIED (proven, 2026-08-16)
`dagger.rollout` records `lam_pol` = the RAW policy output (pre-EMA, pre-mix); the buzz metric + the
show_diag plots are on THAT. Deploy (`deploy_prdot`) plots the EMA-smoothed APPLIED lambda. So the same
net LOOKS buzzier in the dagger diag than in deploy — a plotting-quantity difference, NOT a bug: proved
`dagger.rollout(beta=0)` == `deploy.run_episode` with plant-velocity divergence = 0.000000 across all
drones/3500 steps. Consequence: the buzz gate (on RAW buzz) does NOT catch drones STOPPING (vmin->0) —
a net can PASS `BUZZ_PASS` and still be cooked (eps violated). Consider gating on vmin too, and don't
trust the raw-lambda diag SHAPE as "better/worse" — read the velocity-norm/vmin panel.
