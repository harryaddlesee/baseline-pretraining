#!/bin/bash
#SBATCH --job-name=pyJob2  
#SBATCH --output=output_%j.txt     
#SBATCH --error=error_%j.txt              
#SBATCH --time=01:00:00             
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                      
#SBATCH --cpus-per-task=2                
#SBATCH --mem=16GB                         
#SBATCH --partition=gpu                   
#SBATCH --gres=gpu:1

export CUDA_LAUNCH_BLOCKING=1

# Run your Python script
export BABYLM_ROOT_DIR=/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/trainDir
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 scripts/general_train.py --setting "BabyLM/exp_strict.py:opt125m_s1"
