#!/bin/bash

#SBATCH -J mysubmission
#SBATCH -p New
#SBATCH -n 1
#SBATCH -t 23:59:00
#SBATCH -o test2.out

module load gnu python

python Test1.py > test2.out
