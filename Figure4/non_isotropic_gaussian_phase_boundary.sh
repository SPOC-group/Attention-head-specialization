#!/bin/bash
#SBATCH --job-name=Figure4_boundaries
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00

set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LAMBDA=0.2
NU_TOTAL=9.0
L=5
H=4
F=2

SEED=0
OUT_DIR="results_phase_boundary"
LOG_DIR="logs_phase_boundary"
PYTHON_SCRIPT="../non_isotropic_gaussian/non_isotropic_gaussian_phase_boundary.py"

delta_min=0.01
delta_max=0.5
N_DELTA=50

ALPHA_MIN=0.01
ALPHA_MAX=6.3
N_ALPHAS=50
ALPHA_TOL=1e-3

FIRST_R=0
N_MC=1024
MAX_ITER=2000
DAMPING=0.8
TOL=1e-6
OPT_TOL=1e-9
BRANCH_STARTS=2

i=$SLURM_ARRAY_TASK_ID

DELTA=$(python -c "print(f'{$delta_min + ($delta_max - $delta_min) * $i / ($N_DELTA - 1):.6f}')")

NAME="non_isotropic_gaussian_phase_boundary_lambda_${LAMBDA}_nutot_${NU_TOTAL}_F${F}_L${L}_H${H}"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"

exec > "$LOG_DIR/$NAME/${NAME}_delta_${DELTA}_seed_${SEED}.out" \
     2> "$LOG_DIR/$NAME/${NAME}_delta_${DELTA}_seed_${SEED}.err"

python -u "$PYTHON_SCRIPT" \
    --nu-total "$NU_TOTAL" \
    --delta "$DELTA" \
    --F "$F" \
    --L "$L" \
    --H "$H" \
    --lam "$LAMBDA" \
    --first-r "$FIRST_R" \
    --alpha-min "$ALPHA_MIN" \
    --alpha-max "$ALPHA_MAX" \
    --n-alphas "$N_ALPHAS" \
    --alpha-tol "$ALPHA_TOL" \
    --seed "$SEED" \
    --n-mc "$N_MC" \
    --max-iter "$MAX_ITER" \
    --damping "$DAMPING" \
    --tol "$TOL" \
    --opt-tol "$OPT_TOL" \
    --branch-starts "$BRANCH_STARTS" \
    --output-dir "$OUT_DIR/$NAME"
