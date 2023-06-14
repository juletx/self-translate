#!/bin/bash
#SBATCH --job-name=lm_eval_xglm_multilingual-mt_xglm
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=../.slurm/lm_eval_xglm_multilingual-mt_xglm.out
#SBATCH --error=../.slurm/lm_eval_xglm_multilingual-mt_xglm.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko1/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

# xglm model names
model_names=(
    "facebook/xglm-564M"
    "facebook/xglm-1.7B"
    "facebook/xglm-2.9B"
    "facebook/xglm-7.5B"
)

# load tasks
source ../tasks.sh

# select tasks
tasks_selected=(
    "xcopa-mt_xglm"
    "xstory_cloze-mt_xglm"
    "pawsx-mt_xglm"
    "xnli-mt_xglm"
)

num_fewshot=0

for model_name in "${model_names[@]}"; do
    for group_name in "${tasks_selected[@]}"; do
        srun python3 ../../lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$model_name,dtype=float16 \
            --tasks ${tasks[${group_name}]} \
            --device cuda \
            --output_path ../../results/xglm/${model_name:9}/${model_name:9}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot}
    done
done
