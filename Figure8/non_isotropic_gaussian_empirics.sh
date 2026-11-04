#!/bin/bash
#SBATCH --job-name=Figure8_empirics
#SBATCH --output=empirics_%A_%a.out
#SBATCH --error=empirics_%A_%a.err
#SBATCH --array=0-199
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=0:30:00

set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LAMBDA=0.2
D=4000
NU_1=5.0
NU_2=4.0
L=5
H=4
F=2
INIT_I=0
OUT_DIR="results_empirics"
LOG_DIR="logs_empirics"
PYTHON_SCRIPT="../non_isotropic_gaussian/non_isotropic_gaussian_empirics.py"

alpha_min=0.01
alpha_max=7.0
N_ALPHA=20
SEEDS=($(seq 0 9))
TASK_ID=$SLURM_ARRAY_TASK_ID

alpha_idx=$((TASK_ID % N_ALPHA))
SEED=${SEEDS[$((TASK_ID / N_ALPHA))]}
ALPHA=$(python -c "print(f'{$alpha_min + ($alpha_max - $alpha_min) * $alpha_idx / ($N_ALPHA - 1):.2f}')")

NAME="non_isotropic_gaussian_empirics_lambda_${LAMBDA}_nu1_${NU_1}_nu2_${NU_2}_F${F}_L${L}_H${H}_D${D}_initI${INIT_I}"
mkdir -p "$OUT_DIR/$NAME" "$LOG_DIR/$NAME"
exec > "$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.out" 2> "$LOG_DIR/$NAME/${NAME}_alpha_${ALPHA}_seed_${SEED}.err"

python -u "$PYTHON_SCRIPT" \
    "$D" "$ALPHA" "$NU_1" "$NU_2" "$F" "$L" "$H" "$LAMBDA" "$INIT_I" "$OUT_DIR" "$SEED"
