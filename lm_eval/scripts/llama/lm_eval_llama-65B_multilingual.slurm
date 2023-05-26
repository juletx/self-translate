#!/bin/bash
#SBATCH --job-name=lm_eval_llama-65B_multilingual
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:6
#SBATCH --output=../.slurm/lm_eval_llama-65B_multilingual.out
#SBATCH --error=../.slurm/lm_eval_llama-65B_multilingual.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

# llama model names
model_names=(
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/7B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/13B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/30B"
    "/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/65B"
)

# load tasks
source ../tasks.sh

# select tasks
tasks_selected=(
    #"xcopa"
    "xstory_cloze"
    #"xwinograd"
    "pawsx"
    "xnli"
)

num_fewshot=0

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        srun python3 ../../lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name,load_in_8bit=True,device_map_option=auto \
            --tasks ${tasks[${group_name}]} \
            --device cuda \
            --output_path ../../results/llama/llama-${model_name:48}/llama-${model_name:48}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot}
    done
done