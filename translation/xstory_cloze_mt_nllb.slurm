#!/bin/bash
#SBATCH --job-name=xstory_cloze_mt_nllb
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mt_nllb.out
#SBATCH --error=.slurm/xstory_cloze_mt_nllb.err

# activate virtual environment
source ../venv2/bin/activate

# nllb model names
nllb_model_names=(
    "facebook/nllb-200-distilled-600M"
    "facebook/nllb-200-distilled-1.3B"
    "facebook/nllb-200-1.3B"
    "facebook/nllb-200-3.3B"
)

# run xstory_cloze_mt_nllb.py for each model
for model_name in "${nllb_model_names[@]}"
do
    srun python3 xstory_cloze_mt_nllb.py $model_name
done