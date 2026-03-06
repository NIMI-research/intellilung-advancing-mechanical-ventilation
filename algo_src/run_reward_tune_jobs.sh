#!/bin/bash
#SBATCH -J Reward_Tune
#SBATCH -o slurm_out/reward_tune/%a.out.%x.%j.%N.log
#SBATCH -D  ./
#SBATCH --exclude=i[8023,8024,8005]
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-24
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=1:30:00



mkdir slurm_out/reward_tune -p

module purge
module load release/23.04
module load GCCcore/10.2.0
module load Anaconda3

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate IntelliLungEUEnv


TQDM_DISABLE=1 WANDB_MODE=disabled PYTHONPATH=${PYTHONPATH}:/ python -u reward_tune/run_experiment.py --experiment_root_path "logs/reward-tune-factored-CQL-2025-01-25 14:36:44.904664" --task_id ${SLURM_ARRAY_TASK_ID}