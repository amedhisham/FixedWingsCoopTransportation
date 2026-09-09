---
name: f2-hpc-migration
description: "F2 HPC migration plan (parallel collection DONE) — remaining steps: Ubuntu venv, Linux FMU re-export, SLURM. Concrete code-side TODOs (2/3/4) staged, doing one at a time."
metadata: 
  node_type: memory
  type: project
  originSessionId: e49cf5f7-ad57-4237-adbd-51d78f2ea4a7
  modified: 2026-09-09T14:02:40.885Z
---

**GOAL: move F2 MAPPO training to uni HPC for real (10-30x) speedup.** The parallel-collection prereq is DONE (2026-09-09): `parallel_collect.py` (multiprocessing Pool, per-worker FMU env, actor-only `collect_chunk`, value batched in `main`), `NUM_WORKERS` flag in mappo.py, smoke-tested. See [[f2-slide-fix-todo]] for the diagram that went stale from that refactor. INFRA context in [[f2-residual-rl-plan]] ("build worker-pool as SLURM prereq" — now built).

**Migration is being done ONE STEP AT A TIME (user's call).** Blockers the user named: (1) rebuild venv on Ubuntu, (2) re-export FMU with Linux support, (3) SLURM.

**STEP 1 — venv on Ubuntu: DONE the prep.** `codes/requirements.txt` generated via pip freeze (45 pkgs). CAVEAT: `torch==2.11.0+cu130` is a CUDA local-tag wheel that won't resolve on plain PyPI/Linux — install torch separately on Ubuntu (CPU build is fine; training is DEVICE="cpu", CPU-bound). Rest (casadi/IPOPT, FMPy, numpy, scipy, gymnasium, pettingzoo, matplotlib) are cross-platform. PIN casadi/IPOPT to the validated version so expert trajectories don't shift.

**STEP 2 (SAVED) — compile the FMU for Linux, NO re-export.** Inspected `Base_Model.fmu` (the one `fmu_plant_env` loads by default) 2026-09-09: binaries = `win64/*.dll` ONLY (won't load on Linux), BUT source is FULLY included (`sources/*.c` + `buildDescription.xml`; Simulink/Embedded Coder RTWCG -> portable self-contained C, no MATLAB runtime). So on Ubuntu just COMPILE it: `sudo apt install build-essential` then `python -m fmpy compile Base_Model.fmu` (adds `binaries/linux64/*.so`, same filename, no code change). Verify: `python -c "import fmpy; fmpy.dump('Base_Model.fmu')"`. Fallback only if compile hits missing symbols: re-export from Simulink with Linux binaries. (three_drones fmu = same, but code uses Base_Model.fmu.)

**STEP 3 (SAVED) — SLURM.** parallel_collect is SINGLE-NODE multiprocessing (shared memory, NOT multi-node) -> request `--nodes=1 --cpus-per-task=<many>`, one fat node. Make `NUM_WORKERS` fall back to `os.environ["SLURM_CPUS_PER_TASK"]` so no per-alloc hand-edit. KEEP mp context = "spawn" (NOT fork: main creates its own FMU env before the Pool; fork would copy that live FMU/IPOPT handle into workers = native-state corruption; spawn re-imports cleanly).

**STEP 4 (SAVED) — HPC run hygiene (batch script `train.slurm`).**
- BLAS OVERSUBSCRIPTION footgun: each worker's numpy/torch BLAS spins a full node-sized thread pool -> N workers x N threads = N^2, can be SLOWER than serial. Set `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. (torch.set_num_threads(1) is already pinned per worker in parallel_collect, but the BLAS env vars are separate.)
- HEADLESS matplotlib: `main()` ends with `plt.show()` + builds figures -> errors/hangs on a headless node. Set `MPLBACKEND=Agg` and switch final `plt.show()` -> `savefig`. Small guard, no-op on Windows.
- All of 3/4's code changes are no-ops on the current Windows run (safe to add anytime before travel to HPC).
