#!/bin/bash
# GPU job in the `lidar` environment (OpenPCDet). Command goes after the script name:
#   sbatch --job-name=pp-eval jobs/gpu_lidar.slurm python test.py ...
#SBATCH --job-name=lidar
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/user/%u/logs/%x.%j.out
##SBATCH --account=YOUR_ACCOUNT_NUMBER

source $SCRATCH/envs/activate_lidar.sh
cd $SLURM_SUBMIT_DIR

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "Running: $@"
"$@"
echo "Finished $(date)"
