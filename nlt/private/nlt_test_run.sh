#!/usr/bin/env bash

set -e

GPU=2
ROOT='/data/vision/billf/intrinsic/nlt'
#ckpt="$ROOT"'/output/train/sss_lr:1e-3_mgm:-1/checkpoints/ckpt-27'
ckpt="$ROOT"'/output/train/inter_lr:1e-3_mgm:-1/checkpoints/ckpt-67'
#ckpt="$ROOT"'/output/train/specular_lr:1e-3_mgm:-1/checkpoints/ckpt-80'
n_obs_batches='2'

export TF_FORCE_GPU_ALLOW_GROWTH=true

PYTHONPATH="$ROOT"/neural-light-transport/third_party/:$PYTHONPATH \
    PYTHONPATH="$ROOT"/neural-light-transport/third_party/xiuminglib/:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT"/neural-light-transport/nlt/nlt_test.py \
    --ckpt="$ckpt" \
    --n_obs_batches="$n_obs_batches" \
    "$@"
