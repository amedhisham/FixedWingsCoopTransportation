# HPC migration — F2 MAPPO training

Moving F2 training (`mappo.py`) from the Windows dev box to Ubuntu / uni HPC for the real speedup.
Parallel collection is already implemented (`parallel_collect.py`, `NUM_WORKERS` in `mappo.py`) and
smoke-tested — the FMU step is the bottleneck and is CPU-bound, so it scales ~linearly with cores.

**This doc has two phases:** **Phase 1 = the CPU-HPC interim run** (Steps 0–5 below — big-batch on the
192-core node to solve the current 2-traj) and **Phase 2 = the full GPU migration** (the real target — the
master plan is the "PHASE 2" section near the bottom). Do them in that order; Phase 1 buys progress while
Phase 2 is built.

Do the steps **one at a time**; each later step is a no-op on the current Windows run, so it's safe to
add early. Status (updated 2026-09-11): **[0] git fixed on Ubuntu (fileMode+autocrlf, tree clean) · [1]
DONE — venv built + all imports/CUDA verified · [2] DONE — FMU compiled for linux64, loads+steps · [3] DONE
— NUM_WORKERS auto-scales from SLURM_CPUS_PER_TASK (reserves 1 core for main) · [4] DONE — headless plots
(plt.show guarded, job-id-tagged savefig) + `train.slurm` written · [5] big-batch config pending (walk through
with user) · [Phase 2] GPU plan documented. CLUSTER (DEI/UniPD): SETUP DONE — repo cloned, venv built, FMU
recompiled, smoke run green on a compute node (~99 s/iter, 8 cpus). See "DEI CLUSTER DEPLOYMENT REFERENCE" below.**

> **UBUNTU VENV (built 2026-09-11):** `~/venvs/fixedwings-linux` (ext4, off the shared NTFS), Python 3.12.3,
> **CUDA torch 2.11.0+cu130** — `torch.cuda.is_available()==True` on the GTX 1650. All project imports green
> (numpy 2.2.6, casadi 3.7.2, scipy, fmpy, gymnasium, pettingzoo, matplotlib). Activate:
> `source ~/venvs/fixedwings-linux/bin/activate`. NOTE: `.bashrc` sources ROS jazzy, which puts
> `/opt/ros/jazzy/...` on `PYTHONPATH`; verified it does NOT hijack the venv's numpy/torch (ROS overlay has
> neither), so imports are correct with or without a clean env.

> **BENCHMARK (2026-09-11, same laptop, unmodified code): Ubuntu is ~28% faster than Windows.**
> Iter time **~85 s on Ubuntu vs ~118 s on Windows** (i7-10750H, 6c/12t, NUM_WORKERS=8, STEPS_PER_ITER=92000)
> — BOTH collection and the PPO update got faster. Likely: lighter Linux multiprocessing overhead + the
> natively-gcc-compiled `.so` beating the shipped `.dll` (collection), and no Windows scheduler/BLAS overhead
> (update). This is a pure OS-switch win, before ANY HPC core-scaling — the big multiplier is still the
> 192-core node (Step 5) and Phase 2 GPU. Also confirms Ctrl-C reaches Python cleanly (Linux-native, no forrtl 200).

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
  `.git/config` is shared, so this applies on both OSes — harmless on Windows (it ignores exec bits).
- **CRLF was the second half of the phantom diff (found 2026-09-11).** Contrary to an earlier assumption here,
  `core.autocrlf` was **NOT** set in this checkout, so after the fileMode fix ~424 files still showed as modified —
  a pure CRLF-vs-LF churn (`git diff --ignore-cr-at-eol` was empty; equal `+N/-N` line counts; working tree is
  CRLF, blobs are LF). The fix, config-only, **no file edits and no commit**:
  ```bash
  git config core.autocrlf input   # normalizes CRLF->LF on read/commit; status went 424 -> 0
  ```
  `input` (not `true`) is the correct setting for ONE physical working tree shared by both OSes: git normalizes
  CRLF->LF when comparing working tree to blobs, so **status stays clean on Windows AND Linux** regardless of which
  OS last wrote the file, and it never lets CRLF sneak into a commit (which is what created this mess originally).
  It's shared config, safe on Windows — the only behavioral change there is that a future `checkout`/reset writes
  LF instead of forcing CRLF (fine for this repo's Python/C/docs). Verify pure line-endings first:
  `git diff --ignore-cr-at-eol --stat -- <file>` should be empty before trusting the fix.
- **Never `git reset --hard` / `git checkout .` to "clear" the phantom diff** — it won't stick until fileMode
  is off and risks real files. Set `core.fileMode false` instead.
- Editing code + running git in the shared folder is fine. The venv and heavy checkpoint I/O should NOT live
  on NTFS (slow + OS-specific) — see Step 1.

---

## Step 1 — venv on Ubuntu

Run from this `codes/` dir (where `requirements.txt` lives). The only trick: install **torch separately**
(the pinned `torch==2.11.0+cu130` is a CUDA wheel that isn't on plain PyPI — it needs the PyTorch CUDA index
URL), then install everything else with the torch lines stripped. `torchvision` is **not used** anywhere
(grep-confirmed 2026-09-11) — skip it.

> **INSTALL CUDA TORCH, NOT CPU-ONLY (decided 2026-09-11).** An earlier draft here said install CPU torch —
> that was only because Phase 1 trains with `DEVICE="cpu"` (FMU-bound, tiny nets → GPU buys nothing for the
> CPU-HPC run) and the CPU wheel is simpler. But **Phase 2 is the GPU batched-plant migration and needs CUDA
> torch**, so install it once now. It's safe everywhere: this dev box has a **GTX 1650 (4 GB, Turing sm_75),
> driver 580 → CUDA 13.0** — exactly matching the `+cu130` pin (no `nvcc`/CUDA toolkit needed; the wheel bundles
> its own runtime, only the driver matters); on a **CPU-only HPC node** CUDA torch just runs in CPU mode
> (`torch.cuda.is_available()==False`, `DEVICE="cpu"` unchanged); on the **A40/3090 GPU nodes** it's what Phase 2
> requires. So Step 5's interim CPU run is unaffected — it still uses `DEVICE="cpu"` regardless of the wheel.

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

# 2. CUDA torch, separately (matches the box's driver: CUDA 13.0 -> cu130). If cu130 wheels aren't up yet,
#    the next-lower cuXXX index that torch publishes is fine (driver is back-compatible); any recent 2.x works.
#    On a CPU-only HPC node this same wheel runs in CPU mode (DEVICE="cpu" unchanged) — no separate CPU install.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
# then sanity-check the GPU is visible on a machine that has one:
python -c "import torch; print('cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(cpu-only node)')"

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

> **DONE 2026-09-11:** compiled with gcc 13.3.0 on the `codes/Base_Model.fmu` copy (that's the one training
> loads — code uses the relative `"Base_Model.fmu"` and runs from `codes/`; note the root-level copy is a
> DIFFERENT file, md5-wise, and was left alone). `fmpy compile` succeeded first try (no missing symbols),
> archive now has BOTH `binaries/win64/Base_Model.dll` and `binaries/linux64/Base_Model.so` (271628→291762 B),
> and it instantiated + stepped on Linux. Win64 binary preserved → still runs on the Windows side.

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

### Template `train.slurm` (DEI cluster — see the DEI DEPLOYMENT section below)
```bash
#!/bin/bash
#SBATCH --job-name=f2_mappo
#SBATCH --ntasks=1                # ONE process; the Pool workers are child procs of it
#SBATCH --cpus-per-task=64        # -> NUM_WORKERS via SLURM_CPUS_PER_TASK (pick per `sinfo` free cores)
#SBATCH --partition=allgroups     # DEI's batch partition (the ONLY one for real jobs)
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=f2_%j.log        # stdout (iter prints) land here; watch with `tail -f`
#SBATCH --error=f2_%j.err
#SBATCH --mail-user=<your DEI email>
#SBATCH --mail-type=END           # END mail includes `seff` efficiency report (calibrate next run)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg             # headless matplotlib (no $DISPLAY on compute nodes)

cd "$SLURM_SUBMIT_DIR"            # submit from codes/ so relative paths (Base_Model.fmu, *.npz) resolve
source ~/venvs/fixedwings-linux/bin/activate   # NO `module load` — DEI has no module system; system py3.12 + venv
srun --unbuffered python -u mappo.py   # --unbuffered (srun) + -u (python) => line-by-line live prints
#                                      #   (without --unbuffered, srun+NFS batch the .log into chunks)
```
> NOTE: DEI uses `--ntasks 1 --cpus-per-task N` (NOT `--nodes`). Our collection is one Python process
> spawning N child worker processes on ONE node — that's a single task with N CPUs, exactly the cluster's
> MATLAB-parfor pattern. To force the 192-core node: `--cpus-per-task` >96 (only runner-11 has it) or
> `--nodelist=runner-11`. GPU (Phase 2): add `--gres=gpu:l40s:N` / `gpu:a40:N` (limits: A40/L40S 8 cores/GPU,
> 3090 6/GPU, 8 GPU/user).

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
- **OS/Windows-code audit (2026-09-11): the code is already cross-platform; nothing to port.** Grepped all
  `.py` for Windows-only imports (`msvcrt`/`winreg`/`win32`/`pywin`/`windll`/`_winapi`) → **none**. The only
  OS-specific bit is the `FOR_DISABLE` env var above (a harmless no-op on Linux). The other signal code —
  `signal.signal(signal.SIGINT, signal.SIG_IGN)` in each worker (`parallel_collect.py:39`) so workers ignore
  Ctrl-C and `main()` alone runs the terminate-and-save cleanup — is **POSIX-native and works better on Linux**
  than Windows (SIGINT reaches Python directly). All other `os.*` calls (`os.getpid`, `os.environ`) are
  cross-platform. So the Ctrl-C / clean-shutdown machinery is fully functional on Ubuntu/HPC as-is.

---

---

# DEI CLUSTER DEPLOYMENT REFERENCE (UniPD) — seeds the future README

Concrete, cluster-specific record of getting F2 training onto the **DEI cluster, University of Padova**.
Docs: https://docs.dei.unipd.it/en/CLUSTER/Overview (+ Slurm_basics, Slurm_GPUs, FAQ, Singularity_basics, vscode).
**Setup verified working 2026-09-11** — venv + FMU + smoke run all green on a compute node.

## Access
- SSH: `ssh deiuser@login.dei.unipd.it` (DEI credentials; `-X` for graphical). Account: https://www.dei.unipd.it/en/account
- **The login node is a GATEWAY ONLY** (its MOTD says so): **no Python, no compiling, no training there.**
  Everything real happens on a compute node via `interactive`/`sinteractive` (setup) or `sbatch` (jobs).
  `git clone`/`rsync` on login are fine (file I/O, not compute).

## Hardware map (→ our phases)
| Node(s) | cores | RAM | GPU | use |
|---|---|---|---|---|
| **runner-11** | **192** (Xeon Platinum 8260) | 6 TB | — | **Phase 1 big-batch** (often full — check `sinfo`) |
| runner-07–10 | 96 | 3 TB | — | **Phase 1 sweet spot** (08/09 often fully idle) |
| runner-01–06 | 48–72 | 1.5–2 TB | runner-04/05/06 have 1× Quadro P2000 (weak, ignore) | small CPU |
| runner-12–19 | 32 | 0.5 TB | — | small CPU |
| gpu5, gpu6 | 64 | 1 TB | **10× A40 (48 GB)** | Phase 2 |
| **gpu7** | 48 (EPYC) | 1.5 TB | **8× L40S (48 GB, 91.6 TFLOPS FP32)** | **Phase 2 best FP32** (~2.5× A40) |
| gpu1–4 | 24–32 | 1–1.5 TB | A40 / RTX 3090 (24 GB) | Phase 2 |

## SLURM basics for this cluster (v24.11.6)
- **Two partitions:** `allgroups` (all real batch/parallel jobs — use this) and `interactive` (setup/debug, <24 h, tiny default).
- **Our job shape:** `--ntasks 1 --cpus-per-task N` — one Python process + N−1 child Pool workers on ONE node
  (single-node multiprocessing; NOT MPI/multi-node). Same pattern as the docs' MATLAB-parfor example.
- **Mandatory sbatch options:** `--ntasks`, `--partition`, `--time`, `--mem`. Time formats: `mm`, `mm:ss`, `hh:mm:ss`, `dd-hh[:mm:ss]`.
- **GPU (Phase 2):** `--gres=gpu:l40s:N` / `gpu:a40:N` / `gpu:rtx:N`. Limits: **8 GPU/user; A40 & L40S 8 cores/GPU; RTX 3090 6 cores/GPU** (In_brief page says 4 for 3090 — assume the stricter). Allocation is exclusive; no per-GPU-mem request.
- **Limits:** 70 running jobs/user; 35-day max per job (request far less to start sooner). **They enforce efficiency** — over-request or idle allocations can get jobs killed. Always set `--mail-type END` (the mail carries the `seff` efficiency report to calibrate the next run).
- **Commands:** `sbatch train.slurm` · `squeue -u $USER` / `squeue -p allgroups` · `squeue --start -j <id>` (est. start) · `scancel <id>` · `scontrol show node runner-11` · live progress `myjobinfo <id>` / `nvtop` / post-hoc `sacct` + `seff <id>`.
- **See free nodes BEFORE submitting** (key column %C = Alloc/Idle/Other/Total):
  ```bash
  sinfo -N -o "%N | %.6t | cores A/I/O/T: %.15C | mem %.9m | gpu %G"
  ```
  `idle~` = free but powered-down (auto-boots on allocation). Requesting all 192 on runner-11 waits for the
  whole node; a queue-friendly 64–96 on runner-08/09 usually starts immediately and still hits ~critical batch.

## Software environment model (NO modules)
- **No Lmod/`module` system, no conda pre-installed.** But **system `python3` = 3.12.13** (also 3.11/3.9/3.8/3.6);
  matches the laptop and the `cp312` torch wheel. **`gcc 8.5.0`** (RHEL 8, glibc ~2.28 — OLDER than the laptop's 2.39).
- **PyPI is reachable from compute nodes** → the **venv+pip path works** (verified). Singularity is the documented
  fallback (build a `.sif` — no root needed on the cluster; prebuilt PyTorch+CUDA images at `/nfsd/opt/sif-images`).
- **Storage:** home is **NFS** (`nas1:/vol/homedir/...`, shared across ALL nodes, ~1.5 TB free) → venv + repo here are
  visible everywhere. `/ext` is **node-local scratch** (temp only, purged) — do NOT put the venv there. Big permanent
  data → group NAS or a Helpdesk ticket.

## The deployment we did (repeatable recipe) — DONE 2026-09-11
Run all of this INSIDE an interactive session, never on login:
```bash
ssh aboulenien@login.dei.unipd.it
git clone <repo> ~/FixedWingsCoopTransportation && cd ~/FixedWingsCoopTransportation && git checkout RL
sinteractive --ntasks 1 --cpus-per-task 8 --time 03:00:00 --mem 16G     # lands on a free compute node
# --- inside the session ---
cd ~/FixedWingsCoopTransportation/codes
python3 -m venv ~/venvs/fixedwings-linux          # venv in NFS home (visible on all nodes)
source ~/venvs/fixedwings-linux/bin/activate
python -m pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130   # CUDA wheel; runs CPU-mode on CPU nodes
grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/reqs.txt && pip install -r /tmp/reqs.txt
python -m fmpy compile Base_Model.fmu             # REBUILD the linux64 .so here (laptop's glibc .so won't load)
python -c "import torch,numpy,scipy,casadi,fmpy,gymnasium,pettingzoo,matplotlib; print('ok',torch.__version__,torch.cuda.is_available())"
```
- **Two gotchas (both from the FAQ) that make this per-machine, not copy-from-laptop:**
  1. **Recompile the FMU on the cluster** — externally-compiled binaries must match cluster libs (glibc 2.28 ≠ laptop 2.39). Verified it instantiates+steps after `fmpy compile`.
  2. **Build the venv on the cluster** — venvs aren't portable; pip works there so just repeat the recipe.
- **Smoke run (runner-06, 8 cpus): full pipeline works** — warmstart loads, 8-worker spawn MP + FMU-in-workers runs,
  Ctrl-C cleanly terminates+saves (Linux-native, no forrtl 200). **~99 s/iter** (vs ~85 s laptop Ubuntu / ~118 s Windows;
  slower only because 8 old-Xeon cores — scales with cores on a bigger allocation).

## Git / sync workflow (laptop ↔ cluster) — the 3 hygiene rules (agreed 2026-09-11)
| direction | what | tool |
|---|---|---|
| laptop → cluster | **code edits** | `git push` (laptop) → `git pull` (cluster) |
| cluster → laptop | checkpoints / plots / logs | `git` (user's choice — accepts the bloat, likes full history) or `rsync` |
| per-machine, NEVER synced | compiled FMU, venv | rebuild locally |

1. **Code changes only on the laptop** → commit → push → `git pull` on the HPC.
2. **The HPC produces results** → commit → push → `git pull` on the laptop.
3. **Editing code on the laptop mid-HPC-run is fine — no stash needed.** Because rules 1&2 keep the two
   sides on **disjoint files** (laptop=code, HPC=results), git auto-merges every time. Sequence on the laptop:
   `git commit` FIRST, then `git pull` (auto-merges any pushed results), then `git push`. Commit-before-pull
   is the habit that never surprises you; stash is only for overlapping edits, which this discipline prevents.
- **Corollary — never git-mutate the HPC tree WHILE a job runs there** (`stash`/`pull`/`checkout`): the live
  job is writing result files; moving them mid-write corrupts them, and the running job already has its code in
  RAM so a pull does nothing for it. Push code from the laptop; sync the HPC only after the job ends — or run
  the new code from a second `git worktree`/clone so the live tree is untouched.
- The `skip-worktree` FMU stays out of this entire dance (never staged, survives pulls).
- **FMU is kept in the repo (one committed baseline) but local recompiles are hidden from git** via
  **`git update-index --skip-worktree codes/Base_Model.fmu`** (set on EACH machine; done on laptop+cluster 2026-09-11).
  `.gitignore` would NOT work (file already tracked). `--skip-worktree` keeps the committed FMU intact, hides the local
  `.so`, survives pulls, and is respected by `git add .`/`-A`/`-u` (won't get staged). Verify: `git ls-files -v codes/Base_Model.fmu` → leading **`S`**. Undo: `--no-skip-worktree`.
- `rsync` results back (run on laptop; login node is fine for transfer):
  ```bash
  rsync -avz --progress aboulenien@login.dei.unipd.it:'~/FixedWingsCoopTransportation/codes/f2_*.log' ./codes/
  ```

## VS Code on the cluster (optional, matches the user's workflow)
From an interactive session: `code tunnel` → open the printed `vscode.dev/tunnel/...` link, log in (GitHub/MS), or use
the local "Remote - Tunnels" extension → "Connect to Tunnel". Edit/run on the cluster with the local UI.

## Watching a job's output / plots
- `sbatch` sends stdout to `--output=f2_%j.log`; run `python -u` so prints appear live; `tail -f f2_<id>.log` to watch.
- `MPLBACKEND=Agg` + `plt.show()`→`savefig(...)` (Step 4 code edit) → plots written as PNG, nothing shown.
