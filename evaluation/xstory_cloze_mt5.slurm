#!/bin/bash
#SBATCH --job-name=xstory_cloze_mt5
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mt5.out
#SBATCH --error=.slurm/xstory_cloze_mt5.err

# activate virtual environment
source ../venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# mt5 model names
model_names=(
    "google/mt5-small"
    "google/mt5-base"
    "google/mt5-large"
    "google/mt5-xl"
    "google/mt5-xxl"
)

# dataset name
dataset_names=(
    "juletxara/xstory_cloze"
    "juletxara/xstory_cloze_mt_nllb-3B"
)

# run xstory_cloze_seq2_seq.py for each dataset and model
for dataset_name in "${dataset_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        srun python3 xstory_cloze_seq2seq.py $model_name $dataset_name
    done
done
