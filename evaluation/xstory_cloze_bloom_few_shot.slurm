#!/bin/bash
#SBATCH --job-name=xstory_cloze_bloom_few_shot
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_bloom_few_shot.out
#SBATCH --error=.slurm/xstory_cloze_bloom_few_shot.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

export CUDA_VISIBLE_DEVICES=0

# bloom model names
model_names=(
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1"
    "bigscience/bloom-1b7"
    "bigscience/bloom-3b"
    "bigscience/bloom-7b1"
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
        srun python3 xstory_cloze_few_shot.py $model_name $dataset_name
    done
done
