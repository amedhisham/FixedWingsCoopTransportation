# HPC migration — F2 MAPPO training

Moving F2 training (`mappo.py`) from the Windows dev box to Ubuntu / uni HPC for the real speedup.
Parallel collection is already implemented (`parallel_collect.py`, `NUM_WORKERS` in `mappo.py`) and
smoke-tested — the FMU step is the bottleneck and is CPU-bound, so it scales ~linearly with cores.

Do the steps **one at a time**; each later step is a no-op on the current Windows run, so it's safe to
add early. Status: **[1] venv prep done · [2][3][4] pending · [5] big-batch config pending.**

> **WHY WE'RE DOING THIS RUN (2026-09-10):** the gradient-noise-scale diagnostic proved F2 residual RL
> is **deeply noise-limited — critical batch ~MILLIONS of steps vs the 92k we run on the laptop** (see
> `claude_memory/f2-noise-limited.md`). The 192-core node collects **~2M steps/iter at ~the SAME
> collection wall-time** the laptop needs for 92k (per-core FMU rate is fixed; 192 workers × ~11.5k ≈
> 2.2M), landing you right at the critical batch. So this CPU run is expected to solve the **current
> 2-traj** case cleanly (the GPU migration is for the trajectory *distribution*, which needs tens of
> millions/iter). **Step 5 below is the config that makes it a big-batch run — the whole point.**

---

## Step 1 — venv on Ubuntu

Run from this `codes/` dir (where `requirements.txt` lives). The only trick: install **CPU torch**
separately (the pinned `torch==2.11.0+cu130` is a CUDA wheel that won't resolve on plain PyPI), then
install everything else with the torch lines stripped. `torchvision` is **not used** anywhere — skip it.

```bash
# 0. system prereq
sudo apt update && sudo apt install -y python3-venv python3-pip

# 1. fresh venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. CPU torch, separately (drop the ==2.11.0 pin if that CPU wheel is missing — any recent 2.x is fine,
#    training is DEVICE="cpu" and uses only core torch/torch.nn)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# 3. everything else, with the two +cu130 torch lines stripped
grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/reqs.txt
pip install -r /tmp/reqs.txt

# 4. verify imports (FMU sim itself won't run until Step 2)
python -c "import torch, numpy, scipy, casadi, fmpy, gymnasium, pettingzoo, matplotlib; print('ok, torch', torch.__version__)"
```

- `casadi` (3.7.2, pinned to the validated version) bundles **IPOPT** in its Linux wheel — no separate
  solver install. Keeping the pin keeps expert trajectories identical.
- `fmpy` installs fine, but can't **instantiate** the FMU until Step 2 — so a full `mappo.py` run fails
  at env creation until the Linux FMU exists. Getting imports green here is the goal of Step 1.

---

## Step 2 — compile the FMU for Linux (NO re-export needed)

`fmu_plant_env.py` loads **`Base_Model.fmu`** by default. Inspected 2026-09-09:
- Binaries: **`binaries/win64/Base_Model.dll` only** — no `linux64`, so it will NOT load on Ubuntu as-is.
- Source: **fully included** (`sources/*.c` + headers + `sources/buildDescription.xml`). It's a Simulink /
  Embedded Coder export (`RTWCG_*`, `tmwtypes.h`, `rtwtypes.h`) → portable, self-contained standard C with
  no MATLAB runtime dependency.

So do NOT re-export from Simulink — just **compile the bundled source into a `linux64` binary on Ubuntu**:

```bash
sudo apt install -y build-essential                 # C toolchain (or `module load gcc` on HPC)
python -m fmpy compile Base_Model.fmu               # adds binaries/linux64/Base_Model.so into the .fmu
```

Then it loads normally (same filename, no code change). Verify before/after:
```bash
python -c "import fmpy; fmpy.dump('Base_Model.fmu')"   # should list a linux64 platform after compile
```

Only residual risk: `fmpy compile` hits a missing symbol/header (unlikely for portable Embedded Coder
source with a `buildDescription.xml`) — test it early; if it fails, THEN fall back to re-exporting from
Simulink with Linux binaries. (`Base_Model_three_drones.fmu` is the same situation if ever used — but the
code uses `Base_Model.fmu`.)

---

## Step 3 — SLURM

`parallel_collect.py` is **single-node** multiprocessing (shared memory, NOT multi-node). So:

- Request one fat node: `--nodes=1 --cpus-per-task=<many>`.
- Auto-detect workers instead of hand-editing `NUM_WORKERS` per allocation — make it fall back to
  `SLURM_CPUS_PER_TASK`:
  ```python
  import os
  NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", NUM_WORKERS))
  ```
- **Keep the multiprocessing context = `"spawn"`** (already set). Do NOT switch to `fork`: `main()`
  creates its own FMU env *before* the Pool, and `fork` would copy that live FMU/IPOPT native handle
  into every worker → state corruption. `spawn` re-imports cleanly (that's why the smoke test was safe).

---

## Step 4 — run hygiene (batch script)

Two footguns to defuse in the SLURM batch script / code:

1. **BLAS oversubscription** — each worker's numpy/torch BLAS defaults to a node-sized thread pool, so
   `N workers × N threads = N²` → can be *slower* than serial. Pin to 1 in the batch script:
   ```bash
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   ```
   (`torch.set_num_threads(1)` is already set per worker in `parallel_collect.py`; the BLAS env vars are
   separate and must be set too.)
2. **Headless matplotlib** — `main()` ends with `plt.show()` + builds figures, which errors/hangs on a
   headless compute node. Set `export MPLBACKEND=Agg` and switch the final `plt.show()` → `savefig(...)`.

### Template `train.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=f2_mappo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16        # -> NUM_WORKERS via SLURM_CPUS_PER_TASK
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=f2_%j.log

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

module load python/3.x            # or whatever the cluster provides
source .venv/bin/activate
srun python -u mappo.py
```

---

## Step 5 — big-batch config (the noise-limited interim CPU run)

The point of the HPC-CPU run is a **near-critical-batch** (~1–2M steps/iter), not just faster 92k. All of
these are `mppo.py` edits; walk through them with the user. Current values noted so you can find the lines.

**5a. Auto-scale `NUM_WORKERS` to the allocation** (same as Step 3 — do it once):
```python
# right after the existing `NUM_WORKERS = 6` line:
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", NUM_WORKERS))
```

**5b. Auto-scale `STEPS_PER_ITER` with the worker count** (current: `STEPS_PER_ITER = 92000`). The laptop
did 8 workers × ~11.5k. Keep the *per-worker* budget and let the batch grow with cores — and keep it a
multiple of the episode length so every worker rolls the same integer number of whole episodes (else you
get the fake-dip imbalance). Episode length ≈ `end_time/dt`; on the ~18 s OVERFIT horizon that's ~1800
steps (VERIFY dt in the env — it was 0.01 s). Replace the constant with a derivation placed **after**
`NUM_WORKERS` is finalized and after `OVERFIT_END`/`end_time` is known (so likely compute it near the top
of `main()` or as a module constant if the horizon is fixed):
```python
EPISODE_STEPS      = 1800        # ≈ end_time/dt for the OVERFIT horizon — VERIFY against the env's dt
EPISODES_PER_WORKER = 8          # keep the laptop's per-worker load; batch then scales with cores
STEPS_PER_ITER = NUM_WORKERS * EPISODES_PER_WORKER * EPISODE_STEPS   # 192 cores -> ~2.76M; 64 -> ~920k
```
(If you'd rather cap the batch, set a smaller `EPISODES_PER_WORKER`, e.g. 4 → ~1.4M on 192 cores. Anything
≥ ~1M puts you in the critical-batch range per `f2-noise-limited.md`.)

**5c. Drop `EPOCHS` for the big batch** (current: `EPOCHS = 8`) — **the most important knob.** A ~2M-step
batch does NOT need 8 reuse passes, and the serial PPO update scales with `EPOCHS × STEPS/ MINIBATCH`, so
8 epochs at 2M ≈ ~10 min/update. Set:
```python
EPOCHS = 2                       # big-batch PPO: few passes. Update ~2 min at 2M instead of ~10.
```

**5d. (Optional) Raise `MINIBATCH_STEPS`** (current: `512`) to cut per-step overhead on the huge batch,
e.g. `MINIBATCH_STEPS = 4096`. Same FLOPs, fewer Python/dispatch iterations. Safe; helps update time.

**5e. (Optional) Bump `LR_ACTOR`** (current: `3e-4`) — a big, low-noise batch tolerates a larger step
(~√-scaling). Try `6e-4`, but watch the log_std floor / entropy (the floor at σ≥0.25 guards the collapse
that killed 6e-4 before). Let `critB`/`gcos` justify it rather than guessing.

**5f. HPC run hygiene** — identical to Step 4 (BLAS `*_NUM_THREADS=1`, `MPLBACKEND=Agg`, `plt.show()` →
`savefig`). Mandatory at high core counts.

**What to expect / how to read it:**
- Iter time ≈ **~93 s collection + ~2 min update ≈ 3.5 min** for a ~2M-step iter (collection wall-time is
  ~constant vs the laptop because per-core rate is fixed; only the update grew, and 5c/5d tame it).
- The diagnostics should now **confirm you escaped the noise floor**: `critB` (see `f2-noise-limited.md`)
  should read **near or below** `STEPS_PER_ITER`, and `gcos` should climb **off ~0** onto a positive,
  coherent value. If so, DET_R should descend **cleanly in far fewer iters** than the laptop's ~300-iter
  noise-crawl (0.539 → target 0.392).
- **Still ignore the first ~20 iters** (EMA warm-up + log_std transient) — same reading rule as on the laptop.
- If the update is still too slow: `EPOCHS = 1`, and/or larger `MINIBATCH_STEPS`. If RAM is a worry it
  isn't — a 2M-step buffer is ~1.5–2 GB, trivial on the node's 1–6 TB.
- `critB` uses `NS=8` contiguous sub-batches; with 192 workers the blocks no longer equal one worker each,
  but contiguous still ≈ groups of whole episodes, so the estimate stays valid (magnitude is
  order-of-magnitude anyway).

**Prereqs before this run:** Steps 1 (venv) → 2 (`fmpy compile Base_Model.fmu`) → 3/4 (SLURM + hygiene),
then submit with `--cpus-per-task` set to the node you want (all 192, or a queue-friendlier 64–96).

---

## Notes
- Claude's saved memory lives under the *Windows* `~/.claude` and won't be visible to a Claude session on
  the Ubuntu side — this file is the portable copy of the plan. On Ubuntu: "continue the HPC migration"
  + point at this file. The full memory set is mirrored in `codes/claude_memory/*.md`.
- The parallel-collection design + rationale (actor-only `collect_chunk`, value batched in `main`, why
  the critic left the rollout) is in the code comments of `mappo.py` / `parallel_collect.py`.
- **`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` is a WINDOWS-only fix — safe to leave, no-op on Linux.** It sits
  at the top of `mappo.py` and `parallel_collect.py` (via `os.environ.setdefault`). On Windows the FMU's
  Intel-Fortran runtime installed a console Ctrl-C handler that `abort()`ed the process (`forrtl: error
  200`) before Python could catch it; this env var disables it. On Linux Ctrl-C already reaches Python
  normally, so the line does nothing — **do NOT remove it** (keeps one codebase for both OSes).
