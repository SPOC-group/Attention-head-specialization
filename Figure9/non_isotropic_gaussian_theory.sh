#!/bin/bash
#SBATCH --job-name=Figure9_theory
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5:00:00

set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LAMBDA=0.2
NU_1=4.0
NU_2=4.0
L=5
H=4
F=2

INIT_I=1
SEED=0
OUT_DIR="results_theory"
LOG_DIR="logs_theory"
PYTHON_SCRIPT="../non_isotropic_gaussian/non_isotropic_gaussian_theory.py"

alpha_min=0.01
alpha_max=7.0
N_ALPHA=100
i=$SLURM_ARRAY_TASK_ID

ALPHA=$(python -c "print(f'{$alpha_min + ($alpha_max - $alpha_min) * $i / ($N_ALPHA - 1):.2f}')")

NAME="non_isotropic_gaussian_theory_lambda_${LAMBDA}_nu1_${NU_1}_nu2_${NU_2}_F${F}_L${L}_H${H}_FREE"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"

exec > "$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.out" 2> "$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.err"

python -u "$PYTHON_SCRIPT" \
    "$ALPHA" "$NU_1" "$NU_2" "$F" "$L" "$H" "$LAMBDA" "$INIT_I" "$OUT_DIR" "$SEED"
