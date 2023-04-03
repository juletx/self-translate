#!/bin/bash
#SBATCH --job-name=xstory_cloze_llama_few_shot
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_llama_few_shot.out
#SBATCH --error=.slurm/xstory_cloze_llama_few_shot.err

# activate virtual environment
source ../../alpaca-lora-mt/venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# llama model names
model_names=(
    "/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/7B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/13B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/30B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/65B"
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
