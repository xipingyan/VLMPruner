#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

reduction_ratio=$1
threshold=$2
token_batch=$3

MY_MODEL_PATH="`pwd`/models/"

export HF_DATASETS_CACHE=$MY_MODEL_PATH
export HF_HOME=$MY_MODEL_PATH
export HUGGINGFACE_HUB_CACHE=$MY_MODEL_PATH
export TRANSFORMERS_CACHE=$MY_MODEL_PATH

CKPT=$MY_MODEL_PATH/llava-v1.5-7b
MODEL=llava-v1.5-7b

GPU_Nums=$(echo $CUDA_VISIBLE_DEVICES | tr -cd '0-9' | wc -m)
echo "GPU_NUM: $GPU_Nums"

save_name="${MODEL}_OCRBench"

python -m llava.eval.model_vqa_ocrbench \
    --model_path $CKPT \
    --image_folder ./playground/data/eval/OCRBench/OCRBench_Images \
    --OCRBench_file ./playground/data/eval/OCRBench/OCRBench/OCRBench.json \
    --output_folder ./playground/data/eval/OCRBench/results  \
    --save_name $save_name \
    --num_workers $GPU_Nums \
    --sparse \
    --attn_implementation sdpa \
    --pruned_layer 2 \
    --image_token_start_index 35 \
    --image_token_length 576 \
    --reduction_ratio $reduction_ratio \
    --pivot_image_token 4 \
    --threshold $threshold \
    --token_batch $token_batch
