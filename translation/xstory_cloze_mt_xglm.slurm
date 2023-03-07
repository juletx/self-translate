#!/bin/bash
#SBATCH --job-name=xstory_cloze_mt_xglm
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mt_xglm.out
#SBATCH --error=.slurm/xstory_cloze_mt_xglm.err

# activate virtual environment
source ../venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# xglm model names
xglm_model_names=(
    "facebook/xglm-564M"
    "facebook/xglm-1.7B"
    "facebook/xglm-2.9B"
    "facebook/xglm-4.5B"
    "facebook/xglm-7.5B"
)

# run xstory_cloze_mt_clm.py for each model
for model_name in "${xglm_model_names[@]}"
do
    srun python3 xstory_cloze_mt_clm.py $model_name
done