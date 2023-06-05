#!/bin/bash
#SBATCH --job-name=translate_few_shot_llama_xnli
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/translate_few_shot_llama_xnli.out
#SBATCH --error=.slurm/translate_few_shot_llama_xnli.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# llama model names
model_names=(
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/7B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/13B"
    "/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/30B"
    #"/gaueko1/hizkuntza-ereduak/LLaMA/lm/huggingface/65B"
)

datasets=(
    "xnli"
)

for dataset in "${datasets[@]}"; do
    if [[ $dataset == "mgsm" ]]; then
        max_new_tokens=128
    else
        max_new_tokens=64
    fi
    for model_name in "${model_names[@]}"; do
        srun accelerate launch --mixed_precision fp16 translate_dataset_few_shot.py \
        --dataset $dataset \
        --target_lang "eng_Latn" \
        --starting_batch_size 128 \
        --model_name $model_name \
        --max_length 1024 \
        --max_new_tokens $max_new_tokens \
        --num_beams 1 \
        --num_return_sequences 1 \
        --precision fp16 \
        --eos_token "\n"
    done
done