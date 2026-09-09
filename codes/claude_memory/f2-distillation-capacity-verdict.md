---
name: f2-distillation-capacity-verdict
description: "DECISIVE (2026-09-01): supervised distillation proves the F2 joint-direction wall is PPO OPTIMIZATION, not capacity — one net fits both single-task expert maps to R^2~0.99. Fix = seed RL from a BC net, don't grow it."
metadata:
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-09-01T11:25:10.664Z
---

**THE QUESTION:** joint RL couldn't fit {+y quintic, +x+y custom} together. Two candidates left after aliasing was ruled out: (A) CAPACITY (net can't REPRESENT both maps) vs (B) PPO OPTIMIZATION (map is representable but policy-gradient can't FIND it). Aliasing is OUT because the **reference trajectory is in the obs** -> the two datasets live in DISJOINT obs regions -> the joint map is a genuine separable function (user's own argument).

**THE PROBE (distill_test.py, 2026-09-01):** roll each single-task EXPERT deterministically on its own traj over N_ROLL desync seeds -> harvest (raw obs -> expert MEAN action) pairs -> SUPERVISED-fit ONE student net to both. PPO removed; supervised fit == the memorization test. ch_y solves +y (harvest 30400 pairs, 0 blow), customxy_ch5 solves +x+y (56000 pairs, 0 blow). SOLO fit = per-expert MSE floor; JOINT = one net both. Widths [(256,256),(512,512)], 400 epochs.

**RESULT — CAPACITY EXONERATED:**
```
              solo R2   joint R2   joint/solo MSE
256: yquintic  0.996  ->  0.989        2.99
     xy_custom 0.995  ->  0.994        1.25
512: yquintic  0.998  ->  0.996        2.06
     xy_custom 0.998  ->  0.998        1.08
```
One net represents BOTH maps to **R^2 ~= 0.99 at 256 wide**. The scary-looking `joint/solo=2.99` is 3x of a NEGLIGIBLE error (joint MSE 0.0014 vs solo 0.0005); R^2 barely moves. **The tell is the width sweep: 256->512 SHRINKS the ratio (2.99->2.06, 1.25->1.08) and lifts joint R^2** — if capacity were binding, more width would be NEEDED and the gap would persist/grow; instead it vanishes. So the residual net CAN hold multi-direction coordination; the barrier is that **PPO can't FIND the representable solution** — off-basin exploration blows up.

**WHY THE JOINT RL FAILED (user's mechanism, 2026-09-01):** the two trajs were BLOWING UP during joint training, and the blowups came LATER -> more accumulated dense punishment (the survive-longer==more-penalty pathology). Same feasibility/incentive trap as the curriculum rungs [[f2-shear-precursor-rewards]]: a dense penalty + fixed -100 terminal makes surviving-then-blowing worse than blowing early -> early-death gradient, no path to the good basin. So "PPO can't find it" is concretely THIS incentive landscape, not a mysterious optimizer failure.

**THE FIX — seed the search, DON'T grow the net:** behavior-clone both experts into one net (exactly this distillation, but KEEP the student), then RL fine-tune FROM it (KL-regularized) so PPO starts INSIDE the good basin instead of exploring into blowups. Immediate confirmation step: BC the joint net + roll it CLOSED-LOOP (scale_test) on +y AND +x+y — open-loop R^2=0.99 proves representable, the closed-loop roll proves the representable solution is STABLE. If it holds both, we effectively already have a multi-direction policy and RL's only job becomes SURPASSING the experts, not discovering them. This closes the loop with [[f2-axis-generalization]] (capacity was inferentially closed; now PROVEN) and [[f2-overfit-adapt-experiment]] (the rungs).

**TOOLING:** distill_test.py — harvest experts over desync seeds, solo-vs-joint fit at WIDTHS, report per-expert MSE + R^2 + joint/solo ratio. `flush=True` added (was block-buffered -> zero visibility till exit; run with `python -u` for live output). Fits are on CPU; `cuda.is_available()=True` and the fit phase (dense batched regression) WOULD benefit from GPU (harvest is FMU-bound, CPU-only). MSE is aggregate over all 10 action dims (dw dominates by magnitude) — a per-head split is available if a verdict is ever ambiguous, but R^2~0.99 per group is unambiguous here.

**Thesis relevance:** clean result — the residual CAN represent multi-direction cooperative control; the wall is RL EXPLORATION on a blowup-prone landscape (scaffolding/curriculum/BC-seeding territory), not representational capacity.
