#!/usr/bin/env bash

set -e
set -x

wts_path='weights/moco_v2_800ep_pretrain.pth.tar'
knn_out_dir='output/knn_1_moco_v2_800ep_r50'
linear_out_dir='output/multiple_linear_1_moco_v2_800ep_r50'

CUDA_VISIBLE_DEVICES=$1 python eval_knn.py \
    -j 16 \
    -b 256 \
    -a resnet_moco \
    --weights "$wts_path" \
    --save "$knn_out_dir" \
    --load_cache \
    /datasets/imagenet/

python save_var_mean.py --x_root "$knn_out_dir"

CUDA_VISIBLE_DEVICES=$1 python eval_multiple_linear.py \
    -p 90 \
    -j 16 \
    -b 256 \
    --arch resnet_moco \
    --weights "$wts_path" \
    --mean_paths "$knn_out_dir"/var_mean.pth.tar \
    --save "$linear_out_dir" \
   /datasets/imagenet/

