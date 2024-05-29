#!/bin/bash

#SBATCH --job-name=singlecputasks
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

# Your script goes here
srun --ntasks=1 echo "I'm task 1"
srun --ntasks=1 echo "I'm task 2"
