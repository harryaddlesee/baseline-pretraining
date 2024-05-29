#!/bin/bash
#SBATCH --job-name=pyJob2  
#SBATCH --output=output_%j.txt     
#SBATCH --error=error_%j.txt              
#SBATCH --time=01:00:00             
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                      
#SBATCH --cpus-per-task=2                
#SBATCH --mem=4GB                         
#SBATCH --partition=gpu                   
#SBATCH --gres=gpu:2                      


# Activate your conda environment
source activate myenv

# Run your Python script
#python /scripts/Test1.py
#srun --cpus-per-task=2 python ./Test1.py &> combined_log_%j.txt
srun python ./Test1.py &> combined_log_%j.txt

# Deactivate the conda environment
conda deactivate
