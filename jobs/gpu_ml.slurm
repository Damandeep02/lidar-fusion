#!/bin/bash
# GPU job in the `ml` environment (used here only for YOLO):
#   sbatch --job-name=yolo jobs/gpu_ml.slurm python run_yolo.py ...
#SBATCH --job-name=yolo
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/user/%u/logs/%x.%j.out
##SBATCH --account=YOUR_ACCOUNT_NUMBER

source $SCRATCH/envs/activate_ml.sh
export YOLO_OFFLINE=True   # no internet on compute nodes; weights were downloaded on the login node
cd $SLURM_SUBMIT_DIR

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "Running: $@"
"$@"
echo "Finished $(date)"
