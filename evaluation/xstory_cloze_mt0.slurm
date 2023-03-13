#!/bin/bash
#SBATCH --job-name=xstory_cloze_mt0
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mt0.out
#SBATCH --error=.slurm/xstory_cloze_mt0.err

# activate virtual environment
source ../venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# mt0 model names
model_names=(
    "bigscience/mt0-small"
    "bigscience/mt0-base"
    "bigscience/mt0-large"
    "bigscience/mt0-xl"
    "bigscience/mt0-xxl"
    "bigscience/mt0-xxl-mt"
    "bigscience/mt0-xxl-p3"
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
