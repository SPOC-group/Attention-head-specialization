#!/bin/bash
#SBATCH --job-name=Figure5_BO
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

NU_1=0.1
NU_2=2.0
L=10
SEED=0

OUT_DIR="results_theory"
LOG_DIR="logs_theory"
PYTHON_SCRIPT="../flipping_spike/flipping_spike_BO.py"

alpha_min=0.01
alpha_max=4.0
N_ALPHA=100
N_MC=20000
BATCH_SIZE=20000
N_QUAD=100

i=$SLURM_ARRAY_TASK_ID

ALPHA=$(python -c "print(f'{$alpha_min + ($alpha_max - $alpha_min) * $i / ($N_ALPHA - 1):.2f}')")

NAME="flipping_spike_BO_nu1_${NU_1}_nu2_${NU_2}_L${L}"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"

LOGFILE="$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.out"
ERRFILE="$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.err"
exec > "$LOGFILE" 2> "$ERRFILE"

python -u "$PYTHON_SCRIPT" \
    "$ALPHA" "$NU_1" "$NU_2" "$L" "$OUT_DIR" "$SEED" \
    --n-mc "$N_MC" --batch-size "$BATCH_SIZE" --n-quad "$N_QUAD"
