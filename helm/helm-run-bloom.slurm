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

function execute {
    # Prints and executes command
    echo $1
    eval "time $1"
}

# Perform dry run with just a single model to download and cache all the datasets
# Override with passed-in CLI arguments
# execute "helm-run --models-to-run openai/davinci openai/code-davinci-001 --dry-run --suite dryrun $* &> dryrun.log"

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
    for group in "${groups[@]}"; do
        # Override with passed-in CLI arguments
        # By default, the command will run the RunSpecs listed in src/helm/benchmark/presentation/run_specs.conf
        # and output results to `benchmark_output/runs/<Today's date e.g., 06-28-2022>`.
        execute "helm-run \
            --models-to-run $model \
            --enable-huggingface-models $model \
            --conf helm/src/helm/benchmark/presentation/run_specs.conf \
            --groups-to-run $group \
            --local \
            --suite v1 \
            --num-threads 4 \
            --max-eval-instances 1000"
    done
done
