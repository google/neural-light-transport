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

"""Script used to generate data/neighbors.

Example Usage:

    python "$ROOT"/neural-light-transport/data_gen/gen_render_params_expects.py \
        --trainvali_cams="$ROOT"'/data/trainvali_cams/*.json' \
        --test_cams="$ROOT"'/data/test_cams/*.json' \
        --trainvali_lights="$ROOT"'/data/trainvali_lights/*.json' \
        --test_lights="$ROOT"'/data/test_lights/*.json' \
        --outdir="$ROOT"/data/neighbors/
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import join, exists
from glob import glob
import json
import numpy as np


parser = ArgumentParser(description="")
parser.add_argument(
    '--trainvali_cams', type=str, required=True,
    help="path to the camera .json used for training/validation")
parser.add_argument(
    '--test_cams', type=str, required=True,
    help="path to the camera .json used for testing")
parser.add_argument(
    '--trainvali_lights', type=str, required=True,
    help="path to the light .json used for training/validation")
parser.add_argument(
    '--test_lights', type=str, required=True,
    help="path to the light .json used for testing")
parser.add_argument(
    '--outdir', type=str, required=True, help="output directory")


def get_neighbors(phys_and_virt, phys):
    neighbors = {}

    for ref in phys_and_virt:
        pos = np.array(ref['position'])

        min_dist = np.inf
        nn_name = None
        for cand in phys:
            cand_pos = np.array(cand['position'])

            dist = np.linalg.norm(pos - cand_pos)
            if dist < min_dist and dist != 0:
                nn_name = cand['name']
                min_dist = dist

        assert nn_name is not None
        neighbors[ref['name']] = nn_name

    return neighbors


def load_jsons(path):
    objs = []
    for json_f in sorted(glob(path)):
        with open(json_f, 'rb') as h:
            obj = json.load(h)
        objs.append(obj)
    return objs


def main(args):
    if not exists(args.outdir):
        makedirs(args.outdir)

    # Load
    trainvali_cams = load_jsons(args.trainvali_cams)
    trainvali_lights = load_jsons(args.trainvali_lights)
    test_cams = load_jsons(args.test_cams)
    test_lights = load_jsons(args.test_lights)

    # Get neighbors
    cam_neighbors = get_neighbors(trainvali_cams + test_cams, trainvali_cams)
    light_neighbors = get_neighbors(
        trainvali_lights + test_lights, trainvali_lights)

    # Dump
    with open(join(args.outdir, 'cams.json'), 'w') as h:
        json.dump(cam_neighbors, h)
    with open(join(args.outdir, 'lights.json'), 'w') as h:
        json.dump(light_neighbors, h)


if __name__ == '__main__':
    main(parser.parse_args())
