#!/bin/bash
#SBATCH --job-name=xstory_cloze_xglm_few_shot
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_xglm_few_shot.out
#SBATCH --error=.slurm/xstory_cloze_xglm_few_shot.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

export CUDA_VISIBLE_DEVICES=7

# xglm model names
model_names=(
    "facebook/xglm-564M"
    "facebook/xglm-1.7B"
    "facebook/xglm-2.9B"
    "facebook/xglm-4.5B"
    "facebook/xglm-7.5B"
)

# dataset name
dataset_names=(
    "xstory_cloze_mt_few_shot"
)

# run xstory_cloze.py for each dataset and model
for dataset_name in "${dataset_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        python3 xstory_cloze_few_shot.py $model_name $dataset_name
    done
done
