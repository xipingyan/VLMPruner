#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE="/cache/huggingface/"
export HF_HOME="/cache/huggingface/"
export HUGGINGFACE_HUB_CACHE="/cache/huggingface/"
export TRANSFORMERS_CACHE="/cache/huggingface/"

model_id="/cache/huggingface/Qwen2-VL-7B-Instruct"
model_name="Qwen2-VL-7B-Instruct"
output_path="./logs/${model_name}/${task}/"
mkdir -p "$output_path"

Sparse=$1
pruned_layer=$2
image_token_start_index=0
image_token_length=0
max_num_trunction=64
reduction_ratio=$3
pivot_image_token=4
pivot_text_token=4


python3 -m accelerate.commands.launch \
    --num_processes=8 \
    --main_process_port 50008 \
    -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=$model_id,max_pixels=1003520,device_map=cuda,use_flash_attention_2=False,Sparse=$Sparse,pruned_layer=$pruned_layer,image_token_start_index=$image_token_start_index,image_token_length=$image_token_length,max_num_trunction=$max_num_trunction,reduction_ratio=$reduction_ratio,pivot_image_token=$pivot_image_token,pivot_text_token=$pivot_text_token \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --output_path "$output_path" \