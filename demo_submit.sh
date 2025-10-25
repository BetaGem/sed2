#!/bin/bash
#SBATCH --job-name=NGC0628
#SBATCH --output=./log/NGC0628_%a.log
#SBATCH --error=./log/NGC0628_%a.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --array=0-11

# load modules and activate conda environment
source ~/.bashrc
conda activate bagpipes

# set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# define parameter arrays
START_VALUES=({0..660..60})
END_VALUES=({60..720..60})

# get current parameters
START=${START_VALUES[$SLURM_ARRAY_TASK_ID]}
END=${END_VALUES[$SLURM_ARRAY_TASK_ID]}

# run tasks
python YOUR_PATH/demo_hpc.py NGC0628 $START $END highres