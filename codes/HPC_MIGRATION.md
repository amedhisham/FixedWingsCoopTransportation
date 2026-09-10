# HPC migration — F2 MAPPO training

Moving F2 training (`mappo.py`) from the Windows dev box to Ubuntu / uni HPC for the real speedup.
Parallel collection is already implemented (`parallel_collect.py`, `NUM_WORKERS` in `mappo.py`) and
smoke-tested — the FMU step is the bottleneck and is CPU-bound, so it scales ~linearly with cores.

**This doc has two phases:** **Phase 1 = the CPU-HPC interim run** (Steps 0–5 below — big-batch on the
192-core node to solve the current 2-traj) and **Phase 2 = the full GPU migration** (the real target — the
master plan is the "PHASE 2" section near the bottom). Do them in that order; Phase 1 buys progress while
Phase 2 is built.

Do the steps **one at a time**; each later step is a no-op on the current Windows run, so it's safe to
add early. Status: **[0] dual-boot notes · [1] venv prep done · [2][3][4] pending · [5] big-batch config pending · [Phase 2] GPU plan documented.**

> **WHY WE'RE DOING THIS RUN (2026-09-10):** the gradient-noise-scale diagnostic proved F2 residual RL
> is **deeply noise-limited — critical batch ~MILLIONS of steps vs the 92k we run on the laptop** (see
> `claude_memory/f2-noise-limited.md`). The 192-core node collects **~2M steps/iter at ~the SAME
> collection wall-time** the laptop needs for 92k (per-core FMU rate is fixed; 192 workers × ~11.5k ≈
> 2.2M), landing you right at the critical batch. So this CPU run is expected to solve the **current
> 2-traj** case cleanly (the GPU migration is for the trajectory *distribution*, which needs tens of
> millions/iter). **Step 5 below is the config that makes it a big-batch run — the whole point.**

---

## Step 0 — dual-boot shared folder (SAME directory on both OSes)

The repo lives on a shared partition (e.g. `E:`) that **both Windows and Ubuntu open as the literally same
directory** — same `.git`, same working tree. Consequences:

- **No `git pull` needed to "get" Windows's commits.** It's the same repo — Ubuntu already has every commit.
  A pull only fetches *new remote* commits; if the remote isn't ahead, it's a no-op.
- **"Lots of changed files" on Ubuntu is a PHANTOM diff, not real changes.** NTFS mounted in Linux exposes
  every file as `0777` (executable), so with `core.fileMode=true` git sees a mode change on *every* file.
  Fix once (this clears it):
  ```bash
  git config core.fileMode false
  git status                    # should go clean
  ```
  `.git/config` is shared, so this applies on both OSes — harmless on Windows (it ignores exec bits). CRLF is
  already handled because `core.autocrlf=true` is set (shared config); if files still show modified after the
  fileMode fix, then also `git config core.autocrlf input` and renormalize.
- **Never `git reset --hard` / `git checkout .` to "clear" the phantom diff** — it won't stick until fileMode
  is off and risks real files. Set `core.fileMode false` instead.
- Editing code + running git in the shared folder is fine. The venv and heavy checkpoint I/O should NOT live
  on NTFS (slow + OS-specific) — see Step 1.

---

## Step 1 — venv on Ubuntu

Run from this `codes/` dir (where `requirements.txt` lives). The only trick: install **CPU torch**
separately (the pinned `torch==2.11.0+cu130` is a CUDA wheel that won't resolve on plain PyPI), then
install everything else with the torch lines stripped. `torchvision` is **not used** anywhere — skip it.

> **DUAL-BOOT: put the venv OUTSIDE the shared folder (on the ext4/Linux side), NOT at `codes/.venv/`.**
> A Windows venv and a Linux venv are different binaries and can't coexist at one path in the shared
> directory (building the Linux one there clobbers the Windows one), and NTFS is slow for the many-small-file
> I/O a venv does. Keep the venv on ext4; work in the shared `codes/` folder as usual. (`.gitignore` still
> ignores `codes/.venv/`, but we're not using that path on Linux.)

```bash
# 0. system prereq
sudo apt update && sudo apt install -y python3-venv python3-pip

# 1. fresh venv OFF the NTFS partition (ext4/home), then work in the shared codes/ dir
python3 -m venv ~/venvs/fixedwings-linux
source ~/venvs/fixedwings-linux/bin/activate
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

---

# PHASE 2 — full GPU migration (the master plan / the real target)

Steps 0–5 above are **Phase 1: the CPU-HPC interim run** — squeeze the biggest batch CPU can give (~2M
steps/iter on the 192-core node) to solve the *current 2-traj* case while we build Phase 2. **Phase 2 is
the actual destination.** Detailed reasoning is in `claude_memory/f2-noise-limited.md` and the
GPU-PLANT FEASIBILITY block of `claude_memory/f2-hpc-migration.md`; this is the consolidated plan.

## Why (the trigger, now proven by measurement)
The gradient-noise-scale diagnostic proved F2 residual RL is **deeply noise-limited — critical batch
~MILLIONS of steps** (see `f2-noise-limited.md`: critB settled in the millions vs the 92k laptop batch,
`gcos≈0` = pure noise-thrash). CPU multiprocessing, even the 192-core node, tops out efficiently in the
**hundreds-of-k/iter** — an order of magnitude short. **2 trajectories already need millions/iter; a
trajectory DISTRIBUTION (the generalization goal) needs far more → thousands of parallel collectors →
only a GPU-tensorized plant delivers that.** This is no longer speculative — the trigger condition ("need
thousands of parallel envs") is met by data, which is also the thesis argument for why the problem needs scale.

## What it is (and is NOT)
- **NOT Isaac Gym / PhysX.** The plant is a custom ODE (fixed-wing aero + cable + suspended load), not an
  articulated rigid body. You do NOT use a physics engine.
- **It IS a hand-written BATCHED-TENSOR env:** the plant dynamics + integrator re-expressed as tensor ops
  with a batch axis = thousands of envs stepped in lockstep on-device, with the WHOLE loop on GPU (F1 base
  net, residual actor/critic, reward, reference library, domain randomization) and **zero CPU↔GPU transfer**.

## The key insight — this is a PLANT PORT ONLY, not a research redesign
The earlier "IPOPT expert is an in-loop wall" fear was WRONG for F2 (verified `residual_marl_env.py:305`
"no CasADi anywhere"). **IPOPT runs only OFFLINE** (trajectory-library generation, once, on CPU); the
**in-loop base is the frozen F1 distilled net** (LocalModelAgent). So there is **no expert to reimplement**
— the port is just the plant. No research gamble.

## What to port (difficulty by component)
| Component | Difficulty |
|---|---|
| **FMU plant dynamics + integrator → batched tensors** | **The real work** — MEDIUM, laborious. Recover eqs from the `.c` / Simulink model (USER built it → guides), re-express as batched tensor math, batched fixed-step RK integrator. |
| F1 base net (LocalModelAgent) + allocation `pinv` | Easy — net is GPU-native; allocation is batched linear algebra |
| Residual actor + critic | Trivial — already torch nets |
| Reward (manifold/stall/load/overspeed/overshoot) | Easy — elementwise/reduction tensor ops |
| Reference library (`traj`, `expert_pos`) | Trivial — precomputed arrays, gather/index on device |
| Domain randomization (sensing noise, AR(1), delay walk) | Easy — noise + ring-buffer gather |

## The two genuinely hard parts
1. **Faithful port + NUMERICAL VALIDATION vs the FMU (the gate).** Auto-generated Embedded Coder `.c` is
   scalar soup — reconstruct the *actual* dynamics and verify divergence against the FMU across the operating
   envelope. A wrong sign/coefficient trains beautifully and means nothing → validation is the gate, not a
   spot-check. USER built the FMU → fast physics review in the loop.
2. **Stiff cable constraint → integrator choice.** Cable-suspended dynamics can be stiff; that dictates
   step size / integrator (fixed-step RK4 batched, watch stability). Surfaces early in the prototype.

## Tooling (kernel-launch discipline is mandatory)
PyTorch (`vmap` / **CUDA graphs**) or **NVIDIA Warp** (compiles fused kernels — ideal for this). The
integrator MUST be vectorized across envs with **few, fused ops per step**; a naive Python loop over 1800
steps firing dozens of tiny kernels goes **launch-bound**, especially on a weak GPU — killing the win.

## Hardware
- **GTX 1650 = dev + validate card** (and a real local speedup at current scale). Constraint is **VRAM (4GB)**
  — the rollout buffer, not env state, is the cap (~2M-step buffer won't fit) → the 1650 is a *current-scale*
  card. Its weakness is a feature: forces efficient, portable code.
- **A40 cluster nodes = production scale** (`gpu5-6` = 64c + 10× A40, 48GB each; `gpu4` = 32c + 5×3090 + 3×A40).
  VRAM stops mattering; run tens of millions of steps/iter → the distribution becomes trainable.

## De-risk FIRST — a 1-day prototype before committing the week
Port **only the plant step + integrator** for a batch of N envs, run it **open-loop**, and:
1. **Match the FMU** on a known trajectory (numerical divergence check), and
2. **Benchmark envs/sec** vs the CPU FMU.
Green-light the full week if it hits **≥5× and matches the FMU**; if it's launch-bound at ~1.5×, switch to
Warp before building the RL loop around it.

## Effort & decision
- **~1 week** with the user guiding the physics (built the FMU) + the validation pass. Bounded engineering,
  **not** a research gamble (no expert redesign).
- **Decision: GO** — the noise-limited evidence is the justification that was missing. Phase 1 (CPU big-batch)
  runs in parallel as the interim; Phase 2 is the real deliverable.

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
