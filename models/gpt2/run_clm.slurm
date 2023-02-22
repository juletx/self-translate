#!/bin/bash
#SBATCH --job-name=run_clm_gpt2-eus-cc100
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/run_clm_gpt2-eus-cc100.out
#SBATCH --error=.slurm/run_clm_gpt2-eus-cc100.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv/bin/activate

# report to wandb
export WANDB_PROJECT=gpt2-eus

configs=(
    #"configs/gpt2-eus-euscrawl.json" # 124M params
    #"configs/gpt2-eus-mc4.json" # 124M params
    "configs/gpt2-eus-cc100.json" # 124M params
)

for config in "${configs[@]}"; do
    srun torchrun --standalone --nproc_per_node=1 run_clm.py $config
done
