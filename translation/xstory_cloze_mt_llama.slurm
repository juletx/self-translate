#!/bin/bash
#SBATCH --job-name=xstory_cloze_mt_llama_few_shot
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mt_llama_few_shot.out
#SBATCH --error=.slurm/xstory_cloze_mt_llama_few_shot.err

# activate virtual environment
source ../../alpaca-lora-mt/venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# llama model names
model_names=(
    "/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/7B"
    "/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/13B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/30B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/65B"
)

# run xstory_cloze_mt_clm_few_shot.py for each model
for model_name in "${model_names[@]}"
do
    srun python3 xstory_cloze_mt_clm_few_shot.py $model_name
done