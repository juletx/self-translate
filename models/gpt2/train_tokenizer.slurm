#!/bin/bash
#SBATCH --job-name=train_tokenizer
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --output=.slurm/train_tokenizer.out
#SBATCH --error=.slurm/train_tokenizer.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv/bin/activate

# model name
model_name="gpt2"

# dataset name
dataset_names=(
    "HiTZ/euscrawl"
    "mc4"
    "cc100"
)

# run train_tokenizer.py for each model and dataset
for dataset_name in "${dataset_names[@]}"
do
    echo "Training tokenizer for $model_name and $dataset_name"
    srun python3 train_tokenizer.py $model_name $dataset_name
done
