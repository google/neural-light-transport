#!/usr/bin/env bash

set -e

WHERE_YOU_INSTALLED_BLENDER='/data/vision/billf/shapetime/new1/software'
ROOT='/data/vision/billf/intrinsic/nlt'

"$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
    --background \
    --python "$ROOT"/neural-light-transport/data_gen/render.py \
    -- \
    --scene="$ROOT"/data/scenes/dragon_inter.blend \
    --cam_json="$ROOT"/data/trainvali_cams/C02.json \
    --light_json="$ROOT"/data/trainvali_lights/L160.json \
    --cam_nn_json="$ROOT"/data/neighbors/cams.json \
    --light_nn_json="$ROOT"/data/neighbors/lights.json \
    --imh='512' \
    --uvs='512' \
    --spp='256' \
    --outdir="$ROOT"/tmp/render/dragon_inter/ \
    --debug='true' \
    --cached_uv_unwrap="$ROOT"/tmp/dragon_uv_unwrap.pickle
exit

"$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
    --background \
    --python "$ROOT"/neural-light-transport/data_gen/render.py \
    -- \
    --scene="$ROOT"/data/scenes/dragon_inter.blend \
    --cam_json="$ROOT"/data/test_cams/c006.json \
    --light_json="$ROOT"/data/test_lights/l006.json \
    --cam_nn_json="$ROOT"/data/neighbors/cams.json \
    --light_nn_json="$ROOT"/data/neighbors/lights.json \
    --imh='512' \
    --uvs='512' \
    --spp='256' \
    --outdir="$ROOT"/tmp/render/dragon_inter/ \
    --debug='true' \
    --cached_uv_unwrap="$ROOT"/tmp/dragon_uv_unwrap.pickle
exit

"$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
    --background \
    --python "$ROOT"/neural-light-transport/data_gen/render.py \
    -- \
    --scene="$ROOT"/data/scenes/dragon_sss.blend \
    --cam_json="$ROOT"/data/trainvali_cams/C28C.json \
    --light_json="$ROOT"/data/trainvali_lights/L140.json \
    --cam_nn_json="$ROOT"/data/neighbors/cams.json \
    --light_nn_json="$ROOT"/data/neighbors/lights.json \
    --imh='512' \
    --uvs='512' \
    --spp='1024' \
    --outdir="$ROOT"/tmp/render/dragon_sss/trainvali_000020852_C28C_L140 \
    --debug='true' \
    --cached_uv_unwrap="$ROOT"/tmp/dragon_uv_unwrap.pickle
exit

"$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
    --background \
    --python "$ROOT"/neural-light-transport/data_gen/render.py \
    -- \
    --scene="$ROOT"/data/scenes/dragon_specular.blend \
    --cam_json="$ROOT"/data/trainvali_cams/C28C.json \
    --light_json="$ROOT"/data/trainvali_lights/L140.json \
    --cam_nn_json="$ROOT"/data/neighbors/cams.json \
    --light_nn_json="$ROOT"/data/neighbors/lights.json \
    --imh='512' \
    --uvs='512' \
    --spp='256' \
    --outdir="$ROOT"/tmp/render/dragon_specular/trainvali_000020852_C28C_L140 \
    --debug='true' \
    --cached_uv_unwrap="$ROOT"/tmp/dragon_uv_unwrap.pickle
exit
