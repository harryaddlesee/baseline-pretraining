#!/bin/bash
#SBATCH --job-name=preJOB
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Run your Python script
#export BABYLM_ROOT_DIR=/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/trainDir
#python scripts/general_train.py --output_dir trainDir/model_checkpoints_newOpt --do_train
#python scripts/opt_train.py --output_dir trainDir/model_checkpoints --do_train
python scripts/dataset5M.py