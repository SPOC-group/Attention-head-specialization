#!/bin/bash
#SBATCH --job-name=Figure2_empirics
#SBATCH --output=empirics_%A_%a.out
#SBATCH --error=empirics_%A_%a.err
#SBATCH --array=0-199
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00

set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LAMBDA=0.2
D=2000
NU_1=0.1
NU_2=2.0
L=10
H=2
INIT_I=0
OUT_DIR="results_empirics"
LOG_DIR="logs_empirics"
PYTHON_SCRIPT="../flipping_spike/flipping_spike_empirics.py"

ALPHAS=($(seq -f "%.2f" 0.01 0.2 4.0))
SEEDS=($(seq 0 9))
TASK_ID=$SLURM_ARRAY_TASK_ID
ALPHA=${ALPHAS[$((TASK_ID % ${#ALPHAS[@]}))]}
SEED=${SEEDS[$((TASK_ID / ${#ALPHAS[@]}))]}

NAME="flipping_spike_empirics_lambda_${LAMBDA}_nu1_${NU_1}_nu2_${NU_2}_L${L}_H${H}_D${D}_initI${INIT_I}"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"
exec > "$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.out" 2> "$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.err"

python -u "$PYTHON_SCRIPT" \
    "$D" "$ALPHA" "$NU_1" "$NU_2" "$L" "$H" "$LAMBDA" "$INIT_I" "$OUT_DIR" "$SEED"
