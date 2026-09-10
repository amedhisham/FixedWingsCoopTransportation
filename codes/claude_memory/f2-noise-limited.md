---
name: f2-noise-limited
description: DECISIVE 2026-09-10 — F2 residual RL is DEEPLY gradient-noise-limited (critical batch ~MILLIONS of steps vs 92k working). Proven via gradient-noise-scale (critB) + cross-iter cosine (gcos) diagnostics. Flips the verdict: GPU-tensorized plant (thousands of parallel collectors) is now NECESSARY, not optional.
metadata:
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-09-10T00:00:00.000Z
---

**THE DECISIVE F2 RESULT (2026-09-10): F2 residual training is GRADIENT-NOISE-LIMITED, critical batch ~MILLIONS of steps.** Settles the long "do we need more steps per iter?" debate with a NUMBER. USER'S ORIGINAL INSTINCT ("each iter points a different way -> need more steps") was RIGHT; my early "EV high so not variance-limited" was WRONG — EV measures the CRITIC fit and is BLIND to POLICY-GRADIENT variance (that's the whole point).

**DIAGNOSTICS ADDED to mappo.py (in the per-iter print):**
- `critB` = gradient NOISE SCALE / critical batch size (McCandlish et al. 2018, arXiv:1812.06162), in STEPS (vs STEPS_PER_ITER). Method: split the batch into NS=8 **CONTIGUOUS** sub-batches (≈ per-worker ≈ different desync/traj realizations — NOT random-step split, which UNDERESTIMATES because steps within an episode are correlated -> every sub-batch = same episode-blend -> tr(Σ) too small). Two-batch solve: |G_big|²=|mean_i g_i|² (signal), |G_small|²=mean_i|g_i|² (signal+noise) -> extrapolate 1/B→0 for |G|² (UNBIASED, B_big need not be huge; but single-iter est is NOISY -> EMA 0.9). `critB N(raw M)` prints EMA and raw.
- `gcos` = cosine of consecutive full-batch mean gradients. ~0 = consecutive updates disagree = noise-thrash; >0 = coherent direction (then bounce=LR overshoot, not noise).
- Both computed on ALL actor params incl. log_std (NOT excluded — user chose to just ignore early iters instead; discussed excluding log_std since deployed policy = MEAN, but declined the code change).

**READING RULES (hard-won):**
- IGNORE the first ~20 iters: EMA warm-up + the log_std TRANSIENT. Warm-start resets log_std to LOG_STD_INIT=-1.0, then it marches to the 0.25 floor -> a PERSISTENT log_std gradient that PADS gcos. Confirmed: gcos 0.63->~0.0 tracked ent 4.0->2.8 over iters 1-18; once entropy settled, gcos collapsed to ~0 (the real task-gradient cosine). So early gcos 0.63 was the log_std artifact (user correctly guessed this), NOT coherence.
- `raw nan` = G2 (est |G|²) came out ≤0 -> signal below the noise floor (sub-gradients cancel: |mean g_i|² < mean|g_i|²/NS). Extreme noise-limited, not a bug; EMA holds its last value that iter.
- critB MAGNITUDE is order-of-magnitude only (the contiguous split folds in 2-traj TASK-MIX variance -> inflates a bit; but that's LEGITIMATE noise more steps reduces, and gcos≈0 confirms noise-limitation INDEPENDENTLY of the split).

**THE NUMBERS (2-traj MIX +x+y+z / +x+y-z, STEPS_PER_ITER=92k, from an OLDER checkpoint known to reach DET_R -0.392 in ~300 iters):** after the ~20-iter transient, critB EMA settled in the **MILLIONS (~2-3M steps, raw spiking 5M/12M/14M/nan)** = ~25-30x the 92k batch. gcos ≈ 0.0 (even negative) = pure noise-thrash. DET_R bounced -0.50/-0.54 with NO descent despite -0.392 being provably reachable = noise-limited RANDOM WALK. The prior 300-iter grind to -0.392 was AVERAGING noise across iters because each iter's 92k batch was ~25x undersized.

**DECISION — GPU MIGRATION IS NOW THE PLAN, NOT OPTIONAL (flips [[f2-hpc-migration]]'s "CPU optimal" verdict).** If 2 trajectories already need ~millions of steps/iter, generalizing over a trajectory DISTRIBUTION needs far more -> need THOUSANDS of parallel collectors -> only a GPU-tensorized plant (Isaac-style batched env) delivers that. CPU multiprocessing (even the 192-core node) tops out efficiently in the hundreds-of-k/iter, an order of magnitude short. The GPU-plant port is PLANT-PORT-ONLY (no in-loop IPOPT — base = frozen F1 net; see [[f2-hpc-migration]] GPU-PLANT FEASIBILITY): feasible bounded engineering, ~1 week with the user (who built the FMU) guiding the physics + FMU-match validation. 1650 = dev/validate card + local speedup at current scale; A40 cluster nodes = production scale. The noise-scale evidence is the JUSTIFICATION that was missing — the trigger condition ("need thousands of parallel envs") is now MET by measurement, not speculation.

**CITATIONS (for the write-up):**
- `critB` (gradient noise scale / critical batch size) = CANONICAL paper: **McCandlish, Kaplan, Amodei et al. (OpenAI), "An Empirical Model of Large-Batch Training", 2018, arXiv:1812.06162.** The tr(Σ)/|G|² quantity + two-batch estimator are straight from it. Cite directly.
- `gcos` (cross-iter full-batch gradient cosine) = NO single canonical paper — it's OUR diagnostic (proposed this session). DON'T fabricate a source. Honest grounding: (a) same SNR decomposition as McCandlish 1812.06162 — gcos ≈ SNR/(1+SNR) for g=G+noise, so cite McCandlish as the theory; (b) closest NAMED relative is the between-TASK (not cross-iter) gradient cosine in **Yu et al., "Gradient Surgery for Multi-Task Learning" (PCGrad), 2020, arXiv:2001.06782** (negative cosine = task conflict) — cite as precedent for gradient-cosine-as-diagnostic, not as the source of our exact metric. Frame gcos as a derived diagnostic grounded in McCandlish + PCGrad precedent.

**WHY THIS IS THESIS-GRADE:** F2 decentralized-coordination residual RL is gradient-noise-limited (critical batch ≫ practical CPU batch) — a quantitative reason the problem is hard and why scale (GPU) materially changes what's trainable. Links: [[f2-mix-curriculum]] (the mix-training context + earlier batch-lever correction), [[f2-hpc-migration]] (GPU-plant feasibility + node specs), [[f2-residual-rl-plan]] (the coordination POMDP this all serves).
