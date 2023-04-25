#!/bin/bash
#SBATCH --job-name=lm_eval_llama_reading_comprehension
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_llama_reading_comprehension.out
#SBATCH --error=.slurm/lm_eval_llama_reading_comprehension.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# llama model names
model_names=(
    "/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/7B"
    "/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/13B"
    "/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/30B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/65B"
)

# load tasks
source tasks.sh

# select tasks
tasks_selected=(
    "reading_comprehension"
)

num_fewshot=0

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        srun python3 lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name,use_accelerate=True \
            --tasks ${tasks[${group_name}]} \
            --device cuda:0 \
            --output_path results/llama-${model_name:48}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot}
    done
done