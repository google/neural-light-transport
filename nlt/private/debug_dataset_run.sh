#!/usr/bin/env bash

export TF_FORCE_GPU_ALLOW_GROWTH=true

ROOT='/data/vision/billf/intrinsic/nlt'

PYTHONPATH="$ROOT"/neural-light-transport/third_party/:$PYTHONPATH \
    PYTHONPATH="$ROOT"/neural-light-transport/third_party/xiuminglib/:$PYTHONPATH \
    python "$ROOT"/neural-light-transport/nlt/debug/dataset.py \
    "$@"
