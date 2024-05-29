#!/bin/bash
#SBATCH --job-name=pyJob2                 # Job name
#SBATCH --output=output_%j.txt            # Standard output and error log (%j is the job ID)
#SBATCH --error=error_%j.txt              # Standard error log (%j is the job ID)
#SBATCH --time=01:00:00                   # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=2                 # Number of CPU cores per task
#SBATCH --mem=4GB                         # Total memory limit
#SBATCH --partition=gpu                   # Partition name (adjust as necessary)
#SBATCH --gres=gpu:2                      # Request 2 GPUs

# Load any required modules
module load anaconda/2020.07

# Activate your conda environment
source activate myenv

# Run your Python script
#python /scripts/Test1.py
#srun --cpus-per-task=2 python ./Test1.py &> combined_log_%j.txt
srun python ./Test1.py &> combined_log_%j.txt

# Deactivate the conda environment
conda deactivate
