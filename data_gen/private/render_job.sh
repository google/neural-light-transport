#!/usr/bin/env bash

set -e

WHERE_YOU_INSTALLED_BLENDER='/data/vision/billf/shapetime/new1/software'
ROOT='/data/vision/billf/intrinsic/nlt'

"$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
    --background \
    --python "$ROOT"/neural-light-transport/data_gen/render.py \
    -- \
    --cached_uv_unwrap="$ROOT"/tmp/dragon_uv_unwrap.pickle \
    "$@"
