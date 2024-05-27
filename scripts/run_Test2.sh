#!/bin/bash
#SBATCH --job-name=pyJob2                 # Job name
#SBATCH --output=output_%j.txt            # Standard output and error log (%j is the job ID)
#SBATCH --error=error_%j.txt              # Standard error log (%j is the job ID)
#SBATCH --time=01:00:00                   # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=4GB                         # Total memory limit
#SBATCH --partition=gpu                   # Partition name (adjust as necessary)

# Load any required modules (if needed)
module load anaconda/2020.07  # Example of loading Anaconda module (adjust as necessary)

# Activate your conda environment
source activate myenv

# Run your Python script
python /scripts/Test1.py

# Deactivate the conda environment
conda deactivate
