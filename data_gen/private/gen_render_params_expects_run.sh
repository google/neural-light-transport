#!/usr/bin/env bash

set -e

ROOT='/data/vision/billf/intrinsic/nlt'

python "$ROOT"/neural-light-transport/data_gen/gen_render_params_expects.py \
    --mode='trainvali+test' \
    --scene="$ROOT"/data/scenes/dragon_sss.blend \
    --trainvali_cams="$ROOT"'/data/trainvali_cams/*.json' \
    --test_cams="$ROOT"'/data/test_cams/*.json' \
    --trainvali_lights="$ROOT"'/data/trainvali_lights/*.json' \
    --test_lights="$ROOT"'/data/test_lights/*.json' \
    --cam_nn_json="$ROOT"/data/neighbors/cams.json \
    --light_nn_json="$ROOT"/data/neighbors/lights.json \
    --imh='512' \
    --uvs='512' \
    --spp='256' \
    --outroot="$ROOT"/data/scenes/dragon_sss_imh512_uvs512_spp256/ \
    --tmpdir="$ROOT"/tmp/sss
exit

python "$ROOT"/neural-light-transport/data_gen/gen_render_params_expects.py \
    --mode='trainvali+test' \
    --scene="$ROOT"/data/scenes/dragon_inter.blend \
    --trainvali_cams="$ROOT"'/data/trainvali_cams/*.json' \
    --test_cams="$ROOT"'/data/test_cams/*.json' \
    --trainvali_lights="$ROOT"'/data/trainvali_lights/*.json' \
    --test_lights="$ROOT"'/data/test_lights/*.json' \
    --cam_nn_json="$ROOT"/data/neighbors/cams.json \
    --light_nn_json="$ROOT"/data/neighbors/lights.json \
    --imh='512' \
    --uvs='512' \
    --spp='256' \
    --outroot="$ROOT"/data/scenes/dragon_inter_imh512_uvs512_spp256/ \
    --tmpdir="$ROOT"/tmp/inter
exit

python "$ROOT"/neural-light-transport/data_gen/gen_render_params_expects.py \
    --mode='test' \
    --scene="$ROOT"/data/scenes/dragon_specular.blend \
    --trainvali_cams="$ROOT"'/data/trainvali_cams/*.json' \
    --test_cams="$ROOT"'/data/test_cams/*.json' \
    --trainvali_lights="$ROOT"'/data/trainvali_lights/*.json' \
    --test_lights="$ROOT"'/data/test_lights/*.json' \
    --cam_nn_json="$ROOT"/data/neighbors/cams.json \
    --light_nn_json="$ROOT"/data/neighbors/lights.json \
    --imh='512' \
    --uvs='512' \
    --spp='256' \
    --outroot="$ROOT"/data/scenes/dragon_specular_imh512_uvs512_spp256/ \
    --tmpdir="$ROOT"/tmp/specular
exit
