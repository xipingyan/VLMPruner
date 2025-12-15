#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

export HF_DATASETS_CACHE="/cache/huggingface/"
export HF_HOME="/cache/huggingface/"
export HUGGINGFACE_HUB_CACHE="/cache/huggingface/"
export TRANSFORMERS_CACHE="/cache/huggingface/"


MODEL=/cache/huggingface/llava-v1.5-7b
CKPT=llava-v1.5-7b

reduction_ratio=$1
threshold=$2
token_batch=$3

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL \
        --question-file ./playground/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder ./playground/data/eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation sdpa \
        --pruned_layer 2 \
        --image_token_start_index 35 \
        --image_token_length 576 \
        --reduction_ratio $reduction_ratio \
        --pivot_image_token 4 \
        --threshold $threshold \
        --token_batch $token_batch  &
done

wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/$CKPT.jsonl

