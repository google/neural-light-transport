#!/usr/bin/env bash

set -e

ROOT='/data/vision/billf/intrinsic/nlt'

python "$ROOT"/neural-light-transport/data_gen/get_neighbors.py \
    --trainvali_cams="$ROOT"'/data/trainvali_cams/*.json' \
    --test_cams="$ROOT"'/data/test_cams/*.json' \
    --trainvali_lights="$ROOT"'/data/trainvali_lights/*.json' \
    --test_lights="$ROOT"'/data/test_lights/*.json' \
    --outdir="$ROOT"/data/neighbors
