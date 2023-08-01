#!/bin/bash
#SBATCH --job-name=translate_mgsm_nllb
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/translate_mgsm_nllb.out
#SBATCH --error=.slurm/translate_mgsm_nllb.err
#SBATCH --mem=64GB

# activate virtual environment
source ../venv2/bin/activate

# transformers cache
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

model_names=(
    "facebook/nllb-200-distilled-600M"
    "facebook/nllb-200-distilled-1.3B"
    "facebook/nllb-200-1.3B"
    "facebook/nllb-200-3.3B"
)

for model_name in "${model_names[@]}"
do
    srun accelerate launch --mixed_precision fp16 ../translate_dataset_nllb.py \
    --dataset mgsm \
    --target_lang "eng_Latn" \
    --starting_batch_size 1536 \
    --model_name $model_name \
    --max_length 128 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --precision fp16
done