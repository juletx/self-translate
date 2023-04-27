#!/bin/bash
#SBATCH --job-name=helm-run-xglm
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/helm-run-xglm.out
#SBATCH --error=.slurm/helm-run-xglm.err

# activate virtual environment
source ../helm-venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

models=(
    #"facebook/xglm-564M"
    #"facebook/xglm-1.7B"
    #"facebook/xglm-2.9B"
    #"facebook/xglm-4.5B"
    "facebook/xglm-7.5B"
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
