#!/bin/bash
#SBATCH -J Hyper_Param_Tune
#SBATCH -o slurm_out/hyper_param_tune/%a.out.%x.%j.%N.log
#SBATCH -D  ./
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-865
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=1:00:00



mkdir slurm_out/hyper_param_tune -p

module purge
module load release/23.04
module load GCCcore/10.2.0
module load Anaconda3

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate IntelliLungEUEnv


TQDM_DISABLE=1 WANDB_MODE=disabled WANDB_DISABLE_SERVICE=true PYTHONPATH=${PYTHONPATH}:/ python -u hyper_param_tune/run_experiment.py --experiment_root_path "/data/cat/ws/muyo366f-walrus/logs/" --task_id ${SLURM_ARRAY_TASK_ID}