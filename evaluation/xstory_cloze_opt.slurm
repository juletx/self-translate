#!/bin/bash
#SBATCH --job-name=xstory_cloze_opt
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_opt.out
#SBATCH --error=.slurm/xstory_cloze_opt.err

# activate virtual environment
source ../venv/bin/activate
# export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# opt model names
model_names=(
    "facebook/opt-125m"
    "facebook/opt-350m"
    "facebook/opt-1.3b"
    "facebook/opt-2.7b"
    "facebook/opt-6.7b"
    #"facebook/opt-13b"
    #"facebook/opt-30b"
    #"facebook/opt-66b"
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
