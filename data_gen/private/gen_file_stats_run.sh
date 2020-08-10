#!/usr/bin/env bash

set -e

ROOT='/data/vision/billf/intrinsic/nlt'

python "$ROOT"/neural-light-transport/data_gen/gen_file_stats.py \
    --data_root="$ROOT"'/data/scenes/dragon_sss_imh512_uvs512_spp256/' \
    --out_json="$ROOT"/data/scenes/dragon_sss_imh512_uvs512_spp256.json
exit

python "$ROOT"/neural-light-transport/data_gen/gen_file_stats.py \
    --data_root="$ROOT"'/data/scenes/dragon_inter_imh512_uvs512_spp256/' \
    --out_json="$ROOT"/data/scenes/dragon_inter_imh512_uvs512_spp256.json
exit

python "$ROOT"/neural-light-transport/data_gen/gen_file_stats.py \
    --data_root="$ROOT"'/data/scenes/dragon_specular_imh512_uvs512_spp256/' \
    --out_json="$ROOT"/data/scenes/dragon_specular_imh512_uvs512_spp256.json
exit
