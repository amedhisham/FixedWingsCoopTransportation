---
name: f1-curate-redesign
description: "How the user wants DAgger data curation REBUILT (deterministic guaranteed quotas, separate filters, hardest-wins) + the train-new-fully/sample-old restructure that's already DONE"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-08-14T16:45:08.804Z
---

User's hard requirements for F1 DAgger data management. Continues [[f1-antimemorization-plan]].

## DONE (2026-08-14): train-new-fully + sample-old restructure
The OLD flow was a disaster (user: "10 beta=0 iters to reach a reasonable net"): each iter
`D_X = concat(old, new)` -> evict to MAX_AGG -> train on the WHOLE masked ~150k. Two bugs:
(1) VERY long training (trains the entire aggregate every iter); (2) uniform eviction could throw
away FRESH data. **NEW flow in `dagger_prdot.main`:** collect this iter's new data SEPARATELY (nX/nY/
nH/nbin); TRAIN on ALL of it FULLY + top up with a sample of the OLD pool to reach `TRAIN_BUDGET`
(new is NEVER subsampled); THEN update the pool (concat new, bound to `MAX_AGG`). Constants:
`MAX_AGG=250_000` (pool reservoir), `TRAIN_BUDGET=200_000` (per-iter train size). New data volume:
1 row/step (NOT per-drone), 3500 steps/traj (T_END=35), 36 trajs/iter (1 default+5 custom+30 quintic)
= ~126k new rows/iter. Old-sample + pool-eviction: uniform when `USE_CURATION=False` (current), else
hardest-first `np.argsort(-D_H)[:n]` (interim). Removed the old `curate`/`cap_indices`/`recency_weights`
calls from dagger.

## TODO: rebuild the CURATION itself (currently OFF; the `keep_probs`/`curate`/`cap_indices` in
`collect_prdot_data` are RETIRED, not to be reused as-is). User's spec — **Why:** the old one starved
us of hard samples and gave no guarantees. **How to apply:**
- DETERMINISTIC, not stochastic. The old per-point Bernoulli survival could drop a HARD point and keep
  an EASY one — unacceptable. Use sorts/top-K + fixed quotas.
- FIXED output size (a budget), not a different count every iter.
- SEPARATE, COMPOSABLE filters — NOT one mushed `p = p_bin x hardness_decile x recency`. That mush let
  a recent-EASY point outrank an old-HARD one. Nested guaranteed quotas instead:
  1. BIN quota (diversity floor): guaranteed e.g. 20%/50%/30% *counts* from bins 1/2/3.
  2. within a bin, HARDNESS (separate): keep the top-X% hardest (deterministic sort).
  3. within that, RECENCY (separate): guarantee Y% most-recent.
  Framing the user gave: "the hardest + most recent Nk, with a guaranteed minimum diversity."
- HARDNESS DOMINATES. Eviction/cap must be hardest-first; **recency is at most a tiebreaker, NEVER the
  cap priority** (recency-priority eviction is exactly how hard data got thrown away).
- The collection-time hardness `H = ||lam - lam_prev||` (in `collect_prdot_data.rollout`) is a WEAK proxy
  (no policy exists at collect time -> can't be expert-vs-policy). Only DAgger's `H=||lam_exp-lam_pol||`
  is true hardness. Revisit the BC-time proxy in the rebuild.
- NO curation in `train_prdot` (the BC step): it's the first fixed dataset, nothing to cap. `CURATE_BC`
  should just be deleted.
- Plan: run first with everything OFF (uniform) to see if curation is even NEEDED before rebuilding.
