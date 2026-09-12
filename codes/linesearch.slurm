#!/bin/bash
# F2 line-search diagnostic on the DEI cluster. Submit from codes/:  sbatch linesearch.slurm
# Tests whether the -0.39 plateau is a ROTATION TAX on a real descent direction (line-search along the
# rotation-averaged net step) or a true optimum. Loads mappo.WARMSTART, runs LS_K real iters, then
# evals DET_R along theta0 + alpha*d. See linesearch.py header for the reading rules.
#
#SBATCH --job-name=f2_linesearch
#SBATCH --ntasks=1                 # ONE process; Pool workers are child procs (single-node MP, not MPI)
#SBATCH --cpus-per-task=94         # 94 = runner-08 max (CPUEfctv). Fast 1.4M collection (~100s/iter on 93 workers).
#SBATCH --nodelist=runner-08       # idle clean node; drop this line if it won't schedule and take any free node.
#SBATCH --partition=allgroups
#SBATCH --mem=96G
#SBATCH --time=00:30:00            # SHORT: 3 iters + ~20 evals. Short time backfills FAST onto an idle node.
#SBATCH --output=ls_%j.log         # watch live:  tail -f ls_<jobid>.log
#SBATCH --error=ls_%j.err
#SBATCH --mail-user=ahmedhishamabdelghaffar.aboulenien@studenti.unipd.it
#SBATCH --mail-type=END

# --- run hygiene (mandatory at high core counts) ---
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # stop N-workers x N-BLAS = N^2 oversubscription
export MPLBACKEND=Agg
export UPDATE_THREADS=32

export LS_STEPS=1400000            # supercritical batch (critB read 300-700k) -> the direction is SIGNAL not noise
export LS_K=3                      # real iters to average rotation out of d_net

cd "$SLURM_SUBMIT_DIR"             # submit from codes/ so relative paths (Base_Model.fmu, *.pt) resolve
source ~/venvs/fixedwings-linux/bin/activate

srun --unbuffered python -u linesearch.py
