#!/bin/bash

#SBATCH --job-name=singlecputasks
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1

# Your script goes here
srun --ntasks=1 echo "I'm task 1"
srun --ntasks=1 echo "I'm task 2"
