#!/bin/bash
#SBATCH --job-name=xstory_cloze_gpt2
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_gpt2.out
#SBATCH --error=.slurm/xstory_cloze_gpt2.err

# activate virtual environment
source ../venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# gpt2 model names
model_names=(
    "gpt2"
    "gpt2-medium"
    "gpt2-large"
    "gpt2-xl"
)

# dataset name
dataset_names=(
    "juletxara/xstory_cloze"
    "juletxara/xstory_cloze_mt_nllb-3B"
)

# run xstory_cloze.py for each dataset and model
for dataset_name in "${dataset_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        srun python3 xstory_cloze.py $model_name $dataset_name
    done
done
