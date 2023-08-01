#!/bin/bash
#SBATCH --job-name=translate_few_shot_llama-2-13b-chat
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/translate_few_shot_llama-2-13b-chat.out
#SBATCH --error=.slurm/translate_few_shot_llama-2-13b-chat.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# llama model names
model_names=(
    "meta-llama/Llama-2-13b-chat-hf"
)


datasets=(
    "xcopa"
    "mgsm"
    "xstory_cloze"
    "pawsx"
    "xnli"
)

for dataset in "${datasets[@]}"; do
    if [[ $dataset == "mgsm" ]]; then
        max_new_tokens=128
    else
        max_new_tokens=64
    fi
    for model_name in "${model_names[@]}"; do
        srun accelerate launch --mixed_precision fp16 ../translate_dataset_few_shot.py \
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