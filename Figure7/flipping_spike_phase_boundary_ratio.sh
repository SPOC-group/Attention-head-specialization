#!/bin/bash
#SBATCH --job-name=Figure7_ratio
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00

set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LAMBDA=0.2
NU_TOTAL=4.0
L=10
SEED=0

delta_min=0.2
delta_max=1.0
N_DELTA=100

ALPHA_MIN=0.01
ALPHA_MAX=6.1
N_ALPHAS=100
ALPHA_TOL=1e-3
STABILITY_TOL=1e-4
N_MC=1024
SAMPLING=sobol
SE_TOL=1e-6
OPT_TOL=1e-10
MAX_ITER=200
MAX_ROOT_ITER=40
PROX_MAX_ITER=120
BATCH_SIZE=128

OUT_DIR="results_flipping_spike_phase_boundary_ratio"
LOG_DIR="logs_phase_boundary_ratio"
PYTHON_SCRIPT="../flipping_spike/flipping_spike_phase_boundary_ratio.py"

i=$SLURM_ARRAY_TASK_ID

DELTA=$(python -c "print(f'{$delta_min + ($delta_max - $delta_min) * $i / ($N_DELTA - 1):.17g}')")

NAME="flipping_spike_phase_boundary_ratio_lambda_${LAMBDA}_nutot_${NU_TOTAL}_L${L}_H2"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"
exec > "$LOG_DIR/$NAME/${NAME}_delta_${DELTA}_seed_${SEED}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${i}.out" \
     2> "$LOG_DIR/$NAME/${NAME}_delta_${DELTA}_seed_${SEED}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${i}.err"

python -u "$PYTHON_SCRIPT" \
    --nu-total "$NU_TOTAL" --delta "$DELTA" --L "$L" --lam "$LAMBDA" \
    --alpha-min "$ALPHA_MIN" --alpha-max "$ALPHA_MAX" \
    --n-alphas "$N_ALPHAS" --alpha-tol "$ALPHA_TOL" \
    --stability-tol "$STABILITY_TOL" --max-root-iter "$MAX_ROOT_ITER" \
    --n-mc "$N_MC" --sampling "$SAMPLING" --seed "$SEED" \
    --tol "$SE_TOL" --opt-tol "$OPT_TOL" --max-iter "$MAX_ITER" \
    --prox-max-iter "$PROX_MAX_ITER" --batch-size "$BATCH_SIZE" \
    --delta-grid "$delta_min" "$delta_max" "$N_DELTA" \
    --output-dir "$OUT_DIR/$NAME"
