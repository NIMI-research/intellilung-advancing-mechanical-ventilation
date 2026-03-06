#!/bin/bash
#SBATCH -J Hyper_Param_Tune
#SBATCH -o slurm_out/hyper_param_tune/%A_%a.out
#SBATCH -D ./
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

mkdir -p slurm_out/hyper_param_tune

module purge
module load release/23.04
module load GCCcore/10.2.0
module load Anaconda3

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"

TQDM_DISABLE=1 \
WANDB_MODE=disabled \
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
python -u hyper_param_tune/run_experiment.py \
  --experiment_root_path "${EXPERIMENT_ROOT}" \
  --task_id "${SLURM_ARRAY_TASK_ID}"