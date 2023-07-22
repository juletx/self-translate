#!/bin/bash
#SBATCH --job-name=lm_eval_llama-2_mgsm-mt-few-shot
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=../.slurm/lm_eval_llama-2_mgsm-mt-few-shot.out
#SBATCH --error=../.slurm/lm_eval_llama-2_mgsm-mt-few-shot.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

# llama-2 model names
model_names=(
    #"meta-llama/Llama-2-7b-hf"
    #"meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
)

# load tasks
source ../tasks.sh

# select tasks
tasks_selected=(
    "mgsm-mt"
)

num_fewshot=8

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        srun python3 ../../lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name \
            --tasks ${group_name}_${model_name:11}_* \
            --device cuda \
            --output_path ../../results/llama-2/${model_name:11}/${model_name:11}_${group_name}-few-shot_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot}
    done
done
