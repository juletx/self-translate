#!/bin/bash
#SBATCH --job-name=helm-dry-run
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:0
#SBATCH --output=.slurm/helm-dry-run.out
#SBATCH --error=.slurm/helm-dry-run.err

# activate virtual environment
source ../helm-venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# Perform dry run with just a single model to download and cache all the datasets
model="facebook/xglm-7.5B"
helm-run \
    --models-to-run $model \
    --enable-huggingface-models $model \
    --conf helm/src/helm/benchmark/presentation/run_specs.conf \
    --local \
    --dry-run \
    --suite dryrun \
    --num-threads 4 \
    --max-eval-instances 1000