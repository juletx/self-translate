#!/bin/bash
#SBATCH --job-name=helm_tutorial
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/helm_tutorial.out
#SBATCH --error=.slurm/helm_tutorial.err

# activate virtual environment
source ../helm-venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# Create a run specs configuration
echo 'entries: [
  {description: "mmlu:subject=anatomy,model=huggingface/gpt2", priority: 1},
  {description: "mmlu:subject=philosophy,model=huggingface/gpt2", priority: 1},
]' > run_specs.conf

# Run benchmark
helm-run --conf-paths run_specs.conf --local --suite v1 --max-eval-instances 10

# Summarize benchmark results
helm-summarize --suite v1

# Create plots
helm-create-plots --suite v1