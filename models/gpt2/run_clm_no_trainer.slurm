#!/bin/bash
#SBATCH --job-name=run_clm_no_trainer
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/run_clm_no_trainer.out
#SBATCH --error=.slurm/run_clm_no_trainer.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv/bin/activate

config_args=(
    ["gpt2-eus-euscrawl"]="n_embd=768,n_head=12,n_layer=12,vocab_size=50307,activation_function='gelu_new'"         # 124M params
    ["gpt2-medium-eus-euscrawl"]="n_embd=1024,n_head=16,n_layer=24,vocab_size=50307,activation_function='gelu_new'" # 350M params
    ["gpt2-large-eus-euscrawl"]="n_embd=1280,n_head=20,n_layer=36,vocab_size=50307,activation_function='gelu_new'"  # 774M params
    ["gpt2-xl-eus-euscrawl"]="n_embd=1600,n_head=25,n_layer=48,vocab_size=50307,activation_function='gelu_new'"     # 1558M params
)

accelerate config

for model in "${!config_args[@]}"; do
    accelerate launch run_clm_no_trainer.py \
        --model_type gpt2 \
        --tokenizer_name HiTZ/$model \
        --dataset_name HiTZ/euscrawl \
        --dataset_config_name default \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --do_eval \
        --output_dir out/$model \
        --config_overrides=${config_args[$model]} \
        --push_to_hub \
        --hub_model_id HiTZ/$model \
        --with_tracking \
        --report_to wandb \
        --seed 1
done
