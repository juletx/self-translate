#!/bin/bash
#SBATCH --job-name=run_clm_gpt2-eus-euscrawl
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm/run_clm_gpt2-eus-euscrawl.out
#SBATCH --error=.slurm/run_clm_gpt2-eus-euscrawl.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate

# set cache directory
# export TRANSFORMERS_CACHE=/gaueko0/transformers_cache/

# report to wandb
export WANDB_PROJECT=gpt2-eus

# resume an existing run in cases of failure
export WANDB_RESUME=auto

# pre-train model on 4 GPUs
srun torchrun --nproc_per_node=4 run_clm.py configs/gpt2-eus-euscrawl.yaml
