#!/bin/bash
#SBATCH --job-name=lm_eval_bloom_mmlu
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_bloom_mmlu.out
#SBATCH --error=.slurm/lm_eval_bloom_mmlu.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# bloom model names
model_names=(
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1"
    "bigscience/bloom-1b7"
    "bigscience/bloom-3b"
    "bigscience/bloom-7b1"
)

# load tasks
source tasks.sh

# select tasks
tasks_selected=(
    "mmlu"
)

num_fewshot=5

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        srun python3 lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name,use_accelerate=True \
            --tasks ${tasks[${group_name}]} \
            --device cuda:0 \
            --output_path results/${model_name:12}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --num_fewshot ${num_fewshot}
    done
done
