#!/bin/bash
#SBATCH --job-name=lm_eval_mpt-7b_mgsm
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=../.slurm/lm_eval_mpt-7b_mgsm.out
#SBATCH --error=../.slurm/lm_eval_mpt-7b_mgsm.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

# xglm model names
model_names=(
    "mosaicml/mpt-7b"
)

# load tasks
source ../tasks.sh

# select tasks
tasks_selected=(
    "mgsm"
)

num_fewshot=8

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        srun python3 ../../lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name,trust_remote_code=True,dtype=bfloat16 \
            --tasks ${tasks[${group_name}]} \
            --device cuda \
            --output_path ../../results/mpt/${model_name:9}/${model_name:9}_${group_name}_${num_fewshot}-shot.json \
            --batch_size 48 \
            --no_cache \
            --num_fewshot ${num_fewshot} \
            --write_out \
            --output_base_path ../../results/mpt/${model_name:9}/
    done
done
