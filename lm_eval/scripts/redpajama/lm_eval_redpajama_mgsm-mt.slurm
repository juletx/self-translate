#!/bin/bash
#SBATCH --job-name=lm_eval_redpajama_mgsm-mt
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=../.slurm/lm_eval_redpajama_mgsm-mt.out
#SBATCH --error=../.slurm/lm_eval_redpajama_mgsm-mt.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

# xglm model names
model_names=(
    "togethercomputer/RedPajama-INCITE-Base-3B-v1"
    "togethercomputer/RedPajama-INCITE-7B-Base"
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
            --tasks ${tasks[${group_name}]} \
            --device cuda \
            --output_path ../../results/redpajama/${model_name:17}/${model_name:17}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot}
    done
done
