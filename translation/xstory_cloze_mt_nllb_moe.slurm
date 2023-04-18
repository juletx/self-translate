#!/bin/bash
#SBATCH --job-name=xstory_cloze_mt_nllb_moe
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mt_nllb_moe.out
#SBATCH --error=.slurm/xstory_cloze_mt_nllb_moe.err

# activate virtual environment
source ../venv2/bin/activate

# transformers cache
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# nllb model names
nllb_model_names=(
    "facebook/nllb-moe-54b"
)

# run xstory_cloze_mt_nllb.py for each model
for model_name in "${nllb_model_names[@]}"
do
    srun python3 xstory_cloze_mt_nllb.py $model_name
done