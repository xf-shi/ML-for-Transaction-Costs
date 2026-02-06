#!/bin/bash
#SBATCH --job-name=empire_gpu
#SBATCH --partition=cornell
#SBATCH --account=cornell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1  # Request 1 GPU (max 8)
#SBATCH --time=12:00:00     # Time limit hrs:min:sec
#SBATCH --output=job_%j.out

# Run your command
srun python3 SingleAgentPipe.py

