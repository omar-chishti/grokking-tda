#!/bin/bash
# SLURM job-array sweep: one task per seed. Submit from the Code/ directory:
#   sbatch --export=EXPERIMENT=tf_mod97_grok src/grokking_tda/orchestration/templates/slurm_array.sbatch
#SBATCH --job-name=grokking-tda
#SBATCH --partition=gpus              # TODO: set to the real Imperial GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4                    # 5 seeds -> 5 array tasks
#SBATCH --output=results/raw/_logs/%x_%A_%a.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
EXPERIMENT="${EXPERIMENT:-tf_mod97_grok}"
SEED="${SLURM_ARRAY_TASK_ID}"

echo "running ${EXPERIMENT} seed=${SEED} on $(hostname)"
uv run gtda-train +experiment="${EXPERIMENT}" seed="${SEED}" train.device=cuda
