#!/bin/bash
#SBATCH --job-name=lm_eval_polylm_mgsm
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=../.slurm/lm_eval_polylm_mgsm.out
#SBATCH --error=../.slurm/lm_eval_polylm_mgsm.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

# polylm model names
model_names=(
    "DAMO-NLP-MT/polylm-1.7b"
    "DAMO-NLP-MT/polylm-13b"
    "DAMO-NLP-MT/polylm-multialpaca-13b"
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
            --model_args pretrained=$model_name \
            --tasks ${tasks[${group_name}]} \
            --device cuda \
            --output_path ../../results/polylm/${model_name:12}/${model_name:12}_${group_name}_${num_fewshot}-shot.json \
            --batch_size auto \
            --no_cache \
            --num_fewshot ${num_fewshot}
    done
done
