#!/bin/bash
#SBATCH --job-name=xstory_cloze_xglm_ppl
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_xglm_ppl.out
#SBATCH --error=.slurm/xstory_cloze_xglm_ppl.err

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

# dataset name
dataset_name="juletxara/xstory_cloze"

# run xstory_cloze.py for each model
for model_name in "${xglm_model_names[@]}"
do
    srun python3 ppl.py $model_name $dataset_name
done
