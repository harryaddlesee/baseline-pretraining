#!/bin/bash
#SBATCH --nodes=1
#SBATCH -- time=00:05:00
#SBATCH --qos=debug
#SBATCH --partition=shas
#SBATCH --ntasks=1
#SBATCH --job-name=pyJob
#SBATCH --output=pyJob.%j.out
module purge

echo "hello"
sleep 30
echo "goodbye"
