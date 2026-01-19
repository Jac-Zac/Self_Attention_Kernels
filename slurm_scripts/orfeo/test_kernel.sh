#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH -t 00:02:00
#SBATCH --job-name=test_cuda_kernel
#SBATCH --output=test_kernel.out

# Load the cuda module
module load cuda

# Compile the code
make cuda VERSION=$VERSION DEBUG=1

echo "Running GPU benchmark on Orfeo with UV"
OMP_PLACES=cores
OMP_PROC_BIND=close

uv run pytest -v
