#!/bin/bash
#SBATCH --job-name=translate_xstory_cloze_bloom_few_shot
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/translate_xstory_cloze_bloom_few_shot.out
#SBATCH --error=.slurm/translate_xstory_cloze_bloom_few_shot.err

# activate virtual environment
source ../venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# bloom model names
model_names=(
    "bigscience/bloom-560m"
    "bigscience/bloom-1b1"
    "bigscience/bloom-1b7"
    "bigscience/bloom-3b"
    "bigscience/bloom-7b1"
)

for model_name in "${model_names[@]}"
do
    srun accelerate launch --mixed_precision fp16 translate_dataset_few_shot.py \
    --dataset xstory_cloze \
    --target_lang "eng_Latn" \
    --starting_batch_size 512 \
    --model_name $model_name \
    --max_length 1024 \
    --max_new_tokens 64 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --precision fp16 \
    --eos_token "\n"
done