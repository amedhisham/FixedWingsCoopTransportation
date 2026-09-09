---
name: f2-slide-fix-todo
description: "TODO: mappo_residual_overview.pptx top-band diagram is stale after the parallel-collection refactor — Critic must move Phase 1 -> Phase 2, buffer drops 'value', state 58 -> 76"
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-09-09T13:08:20.439Z
---

**TODO (flagged 2026-09-09, deferred): fix the MAPPO diagram in `codes/mappo_residual_overview.pptx`** (built by `scratchpad/make_slide.py`). The TOP band is now WRONG after two code changes:

1. **Critic moved out of collection.** The slide shows Actor + Critic as PARALLEL readers of the env during Phase 1 COLLECT, with the Critic computing `value` per step into the buffer. That's no longer how it works: `collect_chunk()` is **actor-only**; the critic value is computed **batched in `main()` AFTER collection** (safe because the critic is frozen during collection — [[f2-parallel-collection]]). So on the slide the **Critic belongs in Phase 2 (UPDATE), not Phase 1**. New Phase-2 flow: `buffer → Critic V(state) (batched) → GAE → PPO`. During Phase 1, the privileged `state` is still stored to the buffer, but NOT valued.

2. **Buffer contents drop `value`.** Buffer per-step tuple is now `obs, act, logp, state, reward, done` (no `value` — computed later in update).

3. **`state (58)` label is stale → `76`.** Critic input grew to 76 when the desired-load-state (18-D `p_d, R_d, v_d, ω_d`) was added ([[f2-residual-rl-plan]] critic-target / cable-swing-ambiguity fix): `state(42) + delays(4) + carrier tgt(12) + desired load state(18) = 76`.

Bottom band (ResidualMARLEnv.step per-drone pipeline) is UNCHANGED and correct — only the top MAPPO band needs the rework. Edit `make_slide.py` lines ~73-122 (the "TOP BAND" section) and regenerate.
