#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export HF_DATASETS_CACHE="/cache/huggingface/"
export HF_HOME="/cache/huggingface/"
export HUGGINGFACE_HUB_CACHE="/cache/huggingface/"
export TRANSFORMERS_CACHE="/cache/huggingface/"

CKPT=/cache/huggingface/llava-v1.5-7b
MODEL=llava-v1.5-7b

reduction_ratio=$1
threshold=$2
token_batch=$3

python -m llava.eval.model_vqa_science \
    --model-path $CKPT \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$MODEL.jsonl \
    --single-pred-prompt \
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

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$MODEL.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${MODEL}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${MODEL}_result.json
