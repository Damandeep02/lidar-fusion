#!/bin/bash
# CPU-only job in the `lidar` environment (for building KITTI info files):
#   sbatch jobs/cpu_lidar.slurm python -m pcdet.datasets.kitti.kitti_dataset ...
#SBATCH --job-name=kitti-infos
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/%u/logs/%x.%j.out
##SBATCH --account=YOUR_ACCOUNT_NUMBER

source $SCRATCH/envs/activate_lidar.sh
cd $SLURM_SUBMIT_DIR

echo "Running: $@"
"$@"
echo "Finished $(date)"
