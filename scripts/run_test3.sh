#!/bin/bash -l
################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/username
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J pyJob
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=00:05:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=1024
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu
################# Part-2 Shell script ####################
#===============================
# Activate Flight Environment
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh
#==============================
# Activate Package Ecosystem
#------------------------------
# e.g.:
# Load the OpenMPI module for access to `mpirun` command
flight env activate gridware
module load mpi/openmpi
if ! command -v mpirun &>/dev/null; then
echo "No mpirun command found, ensure that a version of MPI is installed and␣
˓→available in PATH" >&2
exit 1
fi
#===========================
# Create results directory
#---------------------------
RESULTS_DIR="$(pwd)/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
#===============================
# Application launch commands
#-------------------------------
# Customize this section to suit your needs.
echo "Executing job commands, current working directory is $(pwd)"
# REPLACE THE FOLLOWING WITH YOUR APPLICATION COMMANDS
echo "Hello, dmog" > $RESULTS_DIR/test.output
echo "This is an example job. It ran on `hostname -s` (as `whoami`)." >> $RESULTS_DIR/
˓→test.output
echo "I was allocated the following GPU devices: $CUDA_VISIBLE_DEVICES" >> $RESULTS_DIR/
˓→test.output
echo "Output file has been generated, please check $RESULTS_DIR/test.output"
