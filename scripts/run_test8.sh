#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --job-name=pyJob
#SBATCH --output=pyJob.%j.out
#SBATCH --partition=gpu
module purge

echo "hello"
sleep 30
echo "goodbye"
