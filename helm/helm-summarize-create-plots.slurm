#!/bin/bash
#SBATCH --job-name=helm-summarize-create-plots
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:0
#SBATCH --output=.slurm/helm-summarize-create-plots.out
#SBATCH --error=.slurm/helm-summarize-create-plots.err

# activate virtual environment
source ../helm-venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# Summarize benchmark results
helm-summarize --suite v1

# Create plots
helm-create-plots --suite v1