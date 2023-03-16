#!/bin/bash
#SBATCH --job-name=xstory_cloze_gpt2_eus
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_gpt2_eus.out
#SBATCH --error=.slurm/xstory_cloze_gpt2_eus.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# gpt2 model names
model_names=(
    "HiTZ/gpt2-eus-euscrawl"
    #"HiTZ/gpt2-eus-mc4"
    #"HiTZ/gpt2-eus-cc100"
    #"HiTZ/gpt2-medium-eus-euscrawl"
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
