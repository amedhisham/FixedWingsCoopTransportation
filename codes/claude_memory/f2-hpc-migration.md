---
name: f2-hpc-migration
description: "F2 HPC migration plan (parallel collection DONE) — remaining steps: Ubuntu venv, Linux FMU re-export, SLURM. Concrete code-side TODOs (2/3/4) staged, doing one at a time."
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-09-09T19:44:09.694Z
---

**GOAL: move F2 MAPPO training to uni HPC for real speedup.** The parallel-collection prereq is DONE (2026-09-09): `parallel_collect.py` (multiprocessing Pool, per-worker FMU env, actor-only `collect_chunk`, value batched in `main`), `NUM_WORKERS` flag in mappo.py. See [[f2-slide-fix-todo]] for the diagram that went stale from that refactor. INFRA context in [[f2-residual-rl-plan]] ("build worker-pool as SLURM prereq" — now built).

**MEASURED (6-core laptop, STEPS_PER_ITER~32k):** iter time by workers: 2w 80s, 3w 70s, 4w 55s, 5w 50s, 6w 43s. Fit `iter ≈ 25 + 111/n` -> **serial floor ~25s (the PPO UPDATE, doesn't parallelize), 1-worker collection ~111s.** Still scaling at 6w -> HPC (16-32 cores) keeps helping toward the ~25s floor (~1.5x more from 6->32). Value-batching refactor also sped BOTH paths (old seq was 195s). Timing split now in the log: `| Xs (coll Y upd Z)`. The 3w dip = WHOLE-EPISODE ROUNDING (episodes 1800 steps) -> on HPC pick STEPS_PER_ITER divisible by `n_workers x 1800`.

**GPU / Isaac verdict (reasoned through 2026-09-09):** DON'T GPU the update — nets are TINY (256x256) so gradient steps are kernel-launch-STARVED on GPU (slower than CPU BLAS); the CPU<->GPU transfer is once/iter and amortized, NOT the issue. GPU only pays if the SIM goes on GPU (Isaac-style thousands of parallel envs) — but that needs REIMPLEMENTING the FMU plant + CasADi expert as GPU tensors (huge, throws away validated physics). AND at CURRENT modest steps/iter you're ~UPDATE-BOUND once cores are plentiful, so GPU sim saves only the residual collection (~marginal). GPU win unlocks ONLY at million-step batches (where update also grows + matmuls saturate the GPU). Verdict: **CPU multiprocessing is optimal for a CPU/FMU plant at current scale.**

**Extra parallel refinements DONE (2026-09-09, in git, user commits):** (a) ship dpos arrays to workers via `build_pairs(dpos_list)` -> NO per-worker CasADi rebuild (was `N_workers x len(MIX_DIRS)` redundant IPOPT rollouts absorbed into iter-1 = the ~97s iter-1); (b) CLEAN Ctrl-C: workers `signal.SIG_IGN`, `ParallelCollector.terminate()` force-kills without join()-hang, pool creation inside the try (old code FROZE the terminal on Ctrl-C -> recover: `taskkill /F /IM python.exe`); (c) coll/upd timing split. Portable memory + `HPC_MIGRATION.md` + `requirements.txt` committed into `codes/claude_memory/` + repo so a fresh Ubuntu Claude has context (RL branch ahead of origin, user pushes).

**Migration is being done ONE STEP AT A TIME (user's call).** Blockers the user named: (1) rebuild venv on Ubuntu, (2) re-export FMU with Linux support, (3) SLURM.

**STEP 1 — venv on Ubuntu: DONE the prep.** `codes/requirements.txt` generated via pip freeze (45 pkgs). CAVEAT: `torch==2.11.0+cu130` is a CUDA local-tag wheel that won't resolve on plain PyPI/Linux — install torch separately on Ubuntu (CPU build is fine; training is DEVICE="cpu", CPU-bound). Rest (casadi/IPOPT, FMPy, numpy, scipy, gymnasium, pettingzoo, matplotlib) are cross-platform. PIN casadi/IPOPT to the validated version so expert trajectories don't shift.

**STEP 2 (SAVED) — compile the FMU for Linux, NO re-export.** Inspected `Base_Model.fmu` (the one `fmu_plant_env` loads by default) 2026-09-09: binaries = `win64/*.dll` ONLY (won't load on Linux), BUT source is FULLY included (`sources/*.c` + `buildDescription.xml`; Simulink/Embedded Coder RTWCG -> portable self-contained C, no MATLAB runtime). So on Ubuntu just COMPILE it: `sudo apt install build-essential` then `python -m fmpy compile Base_Model.fmu` (adds `binaries/linux64/*.so`, same filename, no code change). Verify: `python -c "import fmpy; fmpy.dump('Base_Model.fmu')"`. Fallback only if compile hits missing symbols: re-export from Simulink with Linux binaries. (three_drones fmu = same, but code uses Base_Model.fmu.)

**STEP 3 (SAVED) — SLURM.** parallel_collect is SINGLE-NODE multiprocessing (shared memory, NOT multi-node) -> request `--nodes=1 --cpus-per-task=<many>`, one fat node. Make `NUM_WORKERS` fall back to `os.environ["SLURM_CPUS_PER_TASK"]` so no per-alloc hand-edit. KEEP mp context = "spawn" (NOT fork: main creates its own FMU env before the Pool; fork would copy that live FMU/IPOPT handle into workers = native-state corruption; spawn re-imports cleanly).

**STEP 4 (SAVED) — HPC run hygiene (batch script `train.slurm`).**
- BLAS OVERSUBSCRIPTION footgun: each worker's numpy/torch BLAS spins a full node-sized thread pool -> N workers x N threads = N^2, can be SLOWER than serial. Set `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. (torch.set_num_threads(1) is already pinned per worker in parallel_collect, but the BLAS env vars are separate.)
- HEADLESS matplotlib: `main()` ends with `plt.show()` + builds figures -> errors/hangs on a headless node. Set `MPLBACKEND=Agg` and switch final `plt.show()` -> `savefig`. Small guard, no-op on Windows.
- All of 3/4's code changes are no-ops on the current Windows run (safe to add anytime before travel to HPC).
