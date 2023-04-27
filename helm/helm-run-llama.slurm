#!/bin/bash
#SBATCH --job-name=helm-run-llama
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/helm-run-llama.out
#SBATCH --error=.slurm/helm-run-llama.err

# activate virtual environment
source ../helm-venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

models=(
    "huggyllama/llama-7b"
    #"huggyllama/llama-13b"
    #"huggyllama/llama-30b"
    #"huggyllama/llama-65b"
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
