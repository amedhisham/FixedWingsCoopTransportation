#!/bin/bash
# F2 MAPPO training on the DEI cluster (UniPD). Submit from the codes/ dir:  sbatch train.slurm
# See HPC_MIGRATION.md "DEI CLUSTER DEPLOYMENT REFERENCE" for node/partition/limit details.
#
#SBATCH --job-name=f2_mappo
#SBATCH --ntasks=1                 # ONE process; the Pool workers are child procs of it (single-node MP, not MPI)
#SBATCH --cpus-per-task=94         # -> NUM_WORKERS = 93. 94 is runner-08's MAX (CPUEfctv=94; SLURM holds
#                                  #   back 2 of the 96). 95+ => "node configuration not available" and the
#                                  #   job silently falls through to runner-11 (192c) instead of pinning here.
#SBATCH --nodelist=runner-08       # idle 96-core node (CPUAlloc=0, ~3TB RAM) -> clean unshared run.
#SBATCH --partition=allgroups      # DEI's batch partition (the only one for real jobs)
#SBATCH --mem=96G                  # big-batch buffer + 93 FMU instances. NB: seff sums RSS across all 93 workers,
#                                  #   so its "79.9G/64G" on job 4960725 is shared-page DOUBLE-COUNTING, not real
#                                  #   pressure (job COMPLETED, no OOM). node has ~3TB; 96G is comfortable headroom.
#SBATCH --time=06:00:00            # wall-clock limit; job KILLED at this. 40 iters ~1.5h + startup -> 2h is enough.
#                                  #   SHORT time = better BACKFILL: SLURM slots you onto an idle node before a
#                                  #   higher-priority reservation, instead of waiting behind it (fixes PD (Priority)).
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
