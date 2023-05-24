#!/bin/bash
#SBATCH --job-name=lm_eval_bloom_xnli-mt
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=../.slurm/lm_eval_bloom_xnli-mt.out
#SBATCH --error=../.slurm/lm_eval_bloom_xnli-mt.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

# bloom model names
model_names=(
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1"
    "bigscience/bloom-1b7"
    "bigscience/bloom-3b"
    "bigscience/bloom-7b1"
)

# load tasks
source ../tasks.sh

# select tasks
tasks_selected=(
    "xnli-mt"
)

num_fewshot=0

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        srun python3 ../../lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name \
            --tasks ${tasks[${group_name}]} \
            --device cuda \
            --output_path ../../results/bloom/${model_name:11}/${model_name:11}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot}
    done
done
