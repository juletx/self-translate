#!/bin/bash
#SBATCH --job-name=xstory_cloze_mt_opt_few_shot
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/xstory_cloze_mt_opt_few_shot.out
#SBATCH --error=.slurm/xstory_cloze_mt_opt_few_shot.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# opt model names
model_names=(
    #"facebook/opt-125m"
    #"facebook/opt-350m"
    #"facebook/opt-1.3b"
    "facebook/opt-2.7b"
    "facebook/opt-6.7b"
    #"facebook/opt-13b"
    #"facebook/opt-30b"
    #"facebook/opt-66b"
)

# run xstory_cloze_mt_clm_few_shot.py for each model
for model_name in "${model_names[@]}"
do
    srun python3 xstory_cloze_mt_clm_few_shot.py $model_name
done