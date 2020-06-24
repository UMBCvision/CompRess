#!/usr/bin/env bash

set -e
set -x


CUDA_VISIBLE_DEVICES=0,1,2,3 python kmeans.py \
    -b 128 \
    -j 10 \
    -a resnet_moco \
    --clusters 16000 \
    --weights weights/moco_v2_800ep_pretrain.pth.tar \
    --save output/kmeans_1_moco_v2_800ep_r50 \
    --load_cache \
    /datasets/imagenet

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_kmeans.py \
    -j 16 \
    -a resnet18 \
    --cos \
    --epochs 100 \
    --batch-size 256 \
    --clusters output/kmeans_1_moco_v2_800ep_r50 \
    --save output/cc_1_moco_v2_800ep_r50 \
    /datasets/imagenet

