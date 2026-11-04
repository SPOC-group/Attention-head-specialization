#!/bin/bash
#SBATCH --job-name=Figure3_boundaries
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

NU_1=0.5
NU_2=2.0
L=10
SEED=0

lambda_min=0.05
lambda_max=0.15
N_LAMBDA=100

ALPHA_MIN=0.01
ALPHA_MAX=1.0
N_ALPHA=100
ALPHA_TOL=0.001
STABILITY_TOL=0.0001
N_MC=1024
SE_TOL=0.000001
OPT_TOL=0.0000000001
MAX_ITER=200

OUT_DIR="results_flipping_spike_phase_boundary"
LOG_DIR="logs_boundary"
PYTHON_SCRIPT="../flipping_spike/flipping_spike_phase_boundary.py"

i=$SLURM_ARRAY_TASK_ID

LAMBDA=$(python -c "print(f'{$lambda_min + ($lambda_max - $lambda_min) * $i / ($N_LAMBDA - 1):.12g}')")

NAME="flipping_spike_phase_boundary_nu1_${NU_1}_nu2_${NU_2}_L${L}_H2"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"

exec > "$LOG_DIR/$NAME/${NAME}_lambda_${LAMBDA}_seed_${SEED}.out" 2> "$LOG_DIR/$NAME/${NAME}_lambda_${LAMBDA}_seed_${SEED}.err"

python -u "$PYTHON_SCRIPT" \
    --nu1 "$NU_1" --nu2 "$NU_2" --L "$L" --lam "$LAMBDA" \
    --alpha-min "$ALPHA_MIN" --alpha-max "$ALPHA_MAX" \
    --n-alphas "$N_ALPHA" --alpha-tol "$ALPHA_TOL" \
    --stability-tol "$STABILITY_TOL" \
    --n-mc "$N_MC" --sampling sobol --seed "$SEED" \
    --tol "$SE_TOL" --opt-tol "$OPT_TOL" --max-iter "$MAX_ITER" \
    --lambda-grid "$lambda_min" "$lambda_max" "$N_LAMBDA" \
    --output-dir "$OUT_DIR/$NAME" --scan-diagnostics
