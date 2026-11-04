#!/bin/bash
#SBATCH --job-name=Figure6_BO
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:20:00

set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NUS=(5.0 4.0)
L=5
SEED=0

OUT_DIR="results_theory"
LOG_DIR="logs_theory"
PYTHON_SCRIPT="../non_isotropic_gaussian/non_isotropic_gaussian_BO.py"

alpha_min=0.01
alpha_max=7.0
N_ALPHA=100
N_MC=20000
BATCH_SIZE=20000

i=$SLURM_ARRAY_TASK_ID

ALPHA=$(python -c "print(f'{$alpha_min + ($alpha_max - $alpha_min) * $i / ($N_ALPHA - 1):.2f}')")

NAME="non_isotropic_gaussian_BO"
for f in "${!NUS[@]}"; do NAME+="_nu$((f + 1))_${NUS[f]}"; done
NAME+="_F${#NUS[@]}_L${L}"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"

LOGFILE="$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.out"
ERRFILE="$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.err"
exec > "$LOGFILE" 2> "$ERRFILE"

python -u "$PYTHON_SCRIPT" \
    "$ALPHA" "$L" "$OUT_DIR" "$SEED" \
    --nus "${NUS[@]}" --n-mc "$N_MC" --batch-size "$BATCH_SIZE"
