#!/bin/bash
#SBATCH --job-name=xstory_cloze_mgpt
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mgpt.out
#SBATCH --error=.slurm/xstory_cloze_mgpt.err

# activate virtual environment
source ../venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# mgpt model names
mgpt_model_names=(
    "sberbank-ai/mGPT"
)

# run xstory_cloze.py for each model
for model_name in "${mgpt_model_names[@]}"
do
    srun python3 xstory_cloze.py $model_name
done
