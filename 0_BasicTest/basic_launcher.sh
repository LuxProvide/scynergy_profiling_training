#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --gpus-per-task=4                  # number of gpu per task
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --time=00:05:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200865                  # project account
#SBATCH --qos=default                      # SLURM qos
#SBATCH --job-name=basic_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

srun nvidia-smi
