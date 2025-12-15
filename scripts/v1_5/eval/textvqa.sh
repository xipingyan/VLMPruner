#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

CKPT=/cache/huggingface/llava-v1.5-7b
MODEL=llava-v1.5-7b

export HF_DATASETS_CACHE="/cache/huggingface/"
export HF_HOME="/cache/huggingface/"
export HUGGINGFACE_HUB_CACHE="/cache/huggingface/"
export TRANSFORMERS_CACHE="/cache/huggingface/"

reduction_ratio=$1
threshold=$2
token_batch=$3

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --sparse \
    --attn_implementation sdpa \
    --pruned_layer 2 \
    --image_token_start_index 35 \
    --image_token_length 576 \
    --reduction_ratio $reduction_ratio \
    --is_textvqa \
    --pivot_image_token 4 \
    --threshold $threshold \
    --token_batch $token_batch

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$MODEL.jsonl
