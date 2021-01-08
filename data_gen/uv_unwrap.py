# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UV unwraps the object of interest in the scene.

Example Usage:

    "$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
        --background \
        --python "$ROOT"/neural-light-transport/data_gen/uv_unwrap.py \
        -- \
        --scene="$ROOT"/data/scenes-v2/dragon_specular.blend \
        --object=object \
        --outpath="$ROOT"/data/scenes-v2/dragon_specular_uv.pickle
"""

from sys import argv
from argparse import ArgumentParser
import pickle as pk

# Blender
import bpy

import xiuminglib as xm


parser = ArgumentParser(description="")
parser.add_argument(
    '--scene', type=str, required=True, help="path to the .blend scene")
parser.add_argument('--object', type=str, default=None, help="object name")
parser.add_argument(
    '--angle_limit', type=float, default=89., help=(
        "angle limit; lower for more projection groups, and higher for less "
        "distortion"))
parser.add_argument(
    '--area_weight', type=float, default=1., help=(
        "area weight used to weight projection vectors; higher for fewer "
        "islands"))
parser.add_argument('--outpath', type=str, required=True, help="output .pickle")


def main(args):
    # Open scene
    xm.blender.scene.open_blend(args.scene)
    obj = bpy.data.objects[args.object]

    # Might be expensive if your model has many vertices
    fi_li_vi_u_v = xm.blender.object.smart_uv_unwrap(
        obj, angle_limit=args.angle_limit, area_weight=args.area_weight)

    # fi_li_vi_u_v is dict[face_index]->2d_array. Each row of the 2D array
    # holds the loop index, vertex index, U coordinate, and V coordinate

    # Force extension
    ext = '.pickle'
    if args.outpath.endswith(ext):
        outpath = args.outpath
    else:
        outpath = args.outpath + ext

    # Write pickle
    with open(outpath, 'wb') as h:
        pk.dump(fi_li_vi_u_v, h)


if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]

    main(parser.parse_args(argv))
