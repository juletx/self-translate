#!/bin/bash
#SBATCH --job-name=evaluate_translations
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/evaluate_translations.out
#SBATCH --error=.slurm/evaluate_translations.err

# activate virtual environment
source ../venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

srun python3 evaluate_translations.py