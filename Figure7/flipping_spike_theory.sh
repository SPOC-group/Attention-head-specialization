#!/bin/bash
#SBATCH --job-name=Figure7_theory
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-299
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5:00:00

set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LAMBDA=0.2
NU_1=2.0
NU_2=2.0
L=10
H=2

INIT_I=1
SEED=0
OUT_DIR="results_theory"
LOG_DIR="logs_theory"
PYTHON_SCRIPT="../flipping_spike/flipping_spike_theory.py"

alpha_min=0.01
alpha_max=6.0
N_ALPHA=100
PHASES=(ps mem spec)

i=$SLURM_ARRAY_TASK_ID

alpha_idx=$((i % N_ALPHA))
phase_idx=$((i / N_ALPHA))
PHASE=${PHASES[$phase_idx]}

ALPHA=$(python -c "print(f'{$alpha_min + ($alpha_max - $alpha_min) * $alpha_idx / ($N_ALPHA - 1):.2f}')")

NAME="flipping_spike_theory_lambda_${LAMBDA}_nu1_${NU_1}_nu2_${NU_2}_L${L}_H${H}_${PHASE^^}"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"

LOGFILE="$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.out"
ERRFILE="$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.err"
exec > "$LOGFILE" 2> "$ERRFILE"

python -u "$PYTHON_SCRIPT" \
    "$ALPHA" "$NU_1" "$NU_2" "$L" "$H" "$LAMBDA" "$INIT_I" "$OUT_DIR" "$SEED" --phase "$PHASE"
