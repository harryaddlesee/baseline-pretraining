#!/bin/bash
#SBATCH --job-name=my_python_job          # Job name
#SBATCH --output=output_%j.txt            # Standard output and error log (%j is the job ID)
#SBATCH --error=error_%j.txt              # Standard error log (%j is the job ID)
#SBATCH --time=01:00:00                   # Wall time limit (hh:mm:ss)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=4GB                         # Total memory limit
#SBATCH --partition=your_partition_name   # Partition name (adjust as necessary)

# Load any required modules (if needed)
module load anaconda/2020.07  # Example of loading Anaconda module (adjust as necessary)

# Activate your conda environment
source activate myenv

# Run your Python script
#python /path/to/your_script.py
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29123 general_train.py --setting "BabyLM/exp_strict.py:opt125m_s1"

# Deactivate the conda environment
conda deactivate
