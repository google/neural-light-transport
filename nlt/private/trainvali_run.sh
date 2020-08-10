#!/usr/bin/env bash

GPU=1

export TF_FORCE_GPU_ALLOW_GROWTH=true

ROOT='/data/vision/billf/intrinsic/nlt'

PYTHONPATH="$ROOT"/neural-light-transport/third_party/:$PYTHONPATH \
    PYTHONPATH="$ROOT"/neural-light-transport/third_party/xiuminglib/:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT"/neural-light-transport/nlt/trainvali.py \
    --config='dragon_inter.ini' \
    "$@"
exit

PYTHONPATH="$ROOT"/neural-light-transport/third_party/:$PYTHONPATH \
    PYTHONPATH="$ROOT"/neural-light-transport/third_party/xiuminglib/:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT"/neural-light-transport/nlt/trainvali.py \
    --config='dragon_sss.ini' \
    "$@"
exit

PYTHONPATH="$ROOT"/neural-light-transport/third_party/:$PYTHONPATH \
    PYTHONPATH="$ROOT"/neural-light-transport/third_party/xiuminglib/:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT"/neural-light-transport/nlt/trainvali.py \
    --config='dragon_specular.ini' \
    "$@"
exit
