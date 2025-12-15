#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE="/cache/huggingface/"
export HF_HOME="/cache/huggingface/"
export HUGGINGFACE_HUB_CACHE="/cache/huggingface/"
export TRANSFORMERS_CACHE="/cache/huggingface/"


CKPT=/cache/huggingface/llava-v1.5-7b
MODEL=llava-v1.5-7b

reduction_ratio=$1
threshold=$2
token_batch=$3

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$MODEL.jsonl \
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
    --token_batch $token_batch

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$MODEL.jsonl
