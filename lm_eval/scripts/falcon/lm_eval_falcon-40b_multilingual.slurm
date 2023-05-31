#!/bin/bash
#SBATCH --job-name=lm_eval_falcon-40b_multilingual
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=../.slurm/lm_eval_falcon-40b_multilingual.out
#SBATCH --error=../.slurm/lm_eval_falcon-40b_multilingual.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4,5

# xglm model names
model_names=(
    "tiiuae/falcon-40b"
)

# load tasks
source ../tasks.sh

# select tasks
tasks_selected=(
    "xcopa"
    "xstory_cloze"
    "xwinograd"
    "pawsx"
    "xnli"
)

num_fewshot=0

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        python3 ../../lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name,trust_remote_code=True,dtype=bfloat16 \
            --tasks ${tasks[${group_name}]} \
            --device cuda \
            --output_path ../../results/falcon/${model_name:7}/${model_name:7}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot} >> ../.slurm/lm_eval_falcon-40b_multilingual.out 2>> ../.slurm/lm_eval_falcon-40b_multilingual.err
    done
done
