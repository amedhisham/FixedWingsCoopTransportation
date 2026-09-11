#!/bin/bash
# F2 MAPPO training on the DEI cluster (UniPD). Submit from the codes/ dir:  sbatch train.slurm
# See HPC_MIGRATION.md "DEI CLUSTER DEPLOYMENT REFERENCE" for node/partition/limit details.
#
#SBATCH --job-name=f2_mappo
#SBATCH --ntasks=1                 # ONE process; the Pool workers are child procs of it (single-node MP, not MPI)
#SBATCH --cpus-per-task=95         # -> NUM_WORKERS = 94 (code reserves 1 core for the main proc). Near-full
#                                  #   96-core node -> higher allocated-core utilization; leaves 1 core headroom.
#SBATCH --partition=allgroups      # DEI's batch partition (the only one for real jobs)
#SBATCH --mem=64G                  # big-batch (~1.3M) buffer + 90 FMU instances; right-size later from the seff report
#SBATCH --time=12:00:00            # wall-clock limit; job is KILLED at this. ITERS=120 @ ~3.5 min ~= 7 h. Raise if needed.
#SBATCH --output=f2_%j.log         # stdout (iter prints) -> here.  watch live:  tail -f f2_<jobid>.log
#SBATCH --error=f2_%j.err          # stderr -> here
#SBATCH --mail-user=ahmedhishamabdelghaffar.aboulenien@studenti.unipd.it   # END mail carries the seff efficiency report
#SBATCH --mail-type=END

# --- run hygiene (mandatory at high core counts) ---
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # stop N-workers x N-BLAS-threads = N^2 oversubscription
export MPLBACKEND=Agg               # headless matplotlib (no $DISPLAY on compute nodes); plt.show() is auto-skipped
export UPDATE_THREADS=32            # main-process PPO update thread count (workers stay pinned to 1). Big minibatch
#                                  #   makes >16 worthwhile; watch NUMA past ~24/socket. Overrides the code default.

cd "$SLURM_SUBMIT_DIR"              # submit from codes/ so relative paths (Base_Model.fmu, *.npz, *.pt) resolve
source ~/venvs/fixedwings-linux/bin/activate    # NO `module load` — DEI has none; system py3.12 + our venv

srun --unbuffered python -u mappo.py   # --unbuffered (srun) + -u (python) => line-by-line live prints in the .log
#                                      #   (without --unbuffered, srun+NFS batch the output into chunks)
