#!/bin/bash
#SBATCH --job-name=helm-run-bloom
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/helm-run-bloom.out
#SBATCH --error=.slurm/helm-run-bloom.err

# activate virtual environment
source ../helm-venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

models=(
    #"bigscience/bloom-560m"
    #"bigscience/bloom-1b1"
    #"bigscience/bloom-1b7"
    #"bigscience/bloom-3b"
    "bigscience/bloom-7b1"
)

groups=(
    "gsm"
    "math_regular"
    "math_chain_of_thougth"
)

for model in "${models[@]}"; do
    helm-run \
        --models-to-run $model \
        --enable-huggingface-models $model \
        --conf helm/src/helm/benchmark/presentation/run_specs.conf \
        --groups-to-run $groups \
        --local \
        --suite v1 \
        --num-threads 4 \
        --max-eval-instances 1000
done
