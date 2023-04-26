#!/bin/bash
#SBATCH --job-name=lm_eval_xglm_mathematical_reasoning_few_shot
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_xglm_mathematical_reasoning_few_shot.out
#SBATCH --error=.slurm/lm_eval_xglm_mathematical_reasoning_few_shot.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# xglm model names
model_names=(
    "facebook/xglm-564M"
    "facebook/xglm-1.7B"
    "facebook/xglm-2.9B"
    "facebook/xglm-4.5B"
    "facebook/xglm-7.5B"
)

# load tasks
source tasks.sh

# select tasks
tasks_selected=(
    "mathematical_reasoning_few_shot"
)

num_fewshot=5

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        srun python3 lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name,use_accelerate=True \
            --tasks ${tasks[${group_name}]} \
            --device cuda:0 \
            --output_path results/${model_name:9}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot}
    done
done
