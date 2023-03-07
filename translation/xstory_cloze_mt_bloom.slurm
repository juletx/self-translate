#!/bin/bash
#SBATCH --job-name=xstory_cloze_mt_bloom
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mt_bloom.out
#SBATCH --error=.slurm/xstory_cloze_mt_bloom.err

# activate virtual environment
source ../venv/bin/activate
# export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# bloom model names
bloom_model_names=(
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1"
    "bigscience/bloom-1b7"
    "bigscience/bloom-3b"
    "bigscience/bloom-7b1"
)

# run xstory_cloze_mt_clm.py for each model
for model_name in "${bloom_model_names[@]}"
do
    srun python3 xstory_cloze_mt_clm.py $model_name
done