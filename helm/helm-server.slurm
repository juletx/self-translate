#!/bin/bash
#SBATCH --job-name=helm-server
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:0
#SBATCH --output=.slurm/helm-server.out
#SBATCH --error=.slurm/helm-server.err

# activate virtual environment
source ../helm-venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# Run server
helm-server