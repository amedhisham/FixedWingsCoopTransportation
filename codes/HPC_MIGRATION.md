# HPC migration — F2 MAPPO training

Moving F2 training (`mappo.py`) from the Windows dev box to Ubuntu / uni HPC for the real speedup.
Parallel collection is already implemented (`parallel_collect.py`, `NUM_WORKERS` in `mappo.py`) and
smoke-tested — the FMU step is the bottleneck and is CPU-bound, so it scales ~linearly with cores.

Do the steps **one at a time**; each later step is a no-op on the current Windows run, so it's safe to
add early. Status: **[1] venv prep done · [2][3][4] pending.**

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

## Notes
- Claude's saved memory lives under the *Windows* `~/.claude` and won't be visible to a Claude session on
  the Ubuntu side — this file is the portable copy of the plan. On Ubuntu: "continue the HPC migration"
  + point at this file.
- The parallel-collection design + rationale (actor-only `collect_chunk`, value batched in `main`, why
  the critic left the rollout) is in the code comments of `mappo.py` / `parallel_collect.py`.
