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

"""This script generates diffuse bases: first an approximated albedo map is
computed by averaging RGB UV maps over all camera-light configurations, next
the albedo UV map is weighted by the light visibility maps (which have
occulusion encoded), and finally the weighted maps are resampled to each camera
view.

Additionally, it globs all the input data and persists their paths as well as
existence to the disk, so that the training pipeline doesn't need to glob
the data or check file existence (potentially very expensive depending on your
filesystem) during training.

Example Usage:

    python "$ROOT"/neural-light-transport/data_gen/postproc.py \
        --data_root="$ROOT"/data/scenes-v2/dragon_specular_imh512_uvs1024_spp256/ \
        --out_json="$ROOT"/data/scenes-v2/dragon_specular_imh512_uvs1024_spp256.json
"""

from argparse import ArgumentParser
from os.path import join, exists, basename, relpath
import numpy as np
from tqdm import tqdm

import xiuminglib as xm

from util import dump_json, remap


parser = ArgumentParser(description="")
parser.add_argument(
    '--data_root', type=str, required=True, help="rendered data root directory")
parser.add_argument(
    '--out_json', type=str, required=True, help="output JSON of file paths")


def main(args):
    # ------ Compute diffuse bases

    # Average all to get UV albedo
    rgb_sum = None
    for config_dir in tqdm(
            xm.os.sortglob(args.data_root, 'trainvali_*'),
            desc="Computing albedo"):
        rgb_path = join(config_dir, 'rgb.png')
        rgb = xm.io.img.load(rgb_path, as_array=True)
        rgb = xm.img.normalize_uint(rgb)
        if rgb_sum is None:
            rgb_sum = np.zeros_like(rgb)
        rgb_sum += rgb
    albedo = rgb_sum / rgb_sum.max()

    for config_dir in tqdm(
            xm.os.sortglob(args.data_root, '*'),
            desc="Computing diffuse bases"):
        # Modulate UV albedo with light visibility
        lvis_path = join(config_dir, 'lvis.png')
        lvis = xm.io.img.load(lvis_path, as_array=True)
        lvis = xm.img.normalize_uint(lvis)
        lvis = np.dstack([lvis] * 3)
        diffuse_uv = albedo * lvis
        diffuse_path = join(config_dir, 'diffuse.png')
        diffuse_uint = xm.io.img.write_arr(diffuse_uv, diffuse_path, clip=True)

        uv2cam_path = join(config_dir, 'uv2cam.npy')
        uv2cam = np.load(uv2cam_path)
        diffuse_camspc_uint = remap(diffuse_uint, uv2cam)
        diffuse_camspc_path = join(config_dir, 'diffuse_camspc.png')
        xm.io.img.write_img(diffuse_camspc_uint, diffuse_camspc_path)

    # ------ Generate file list

    gen_file_list(args)


def gen_file_list(args):
    filelist = {}

    for config_dir in tqdm(
            xm.os.sortglob(args.data_root, '*'), desc="Generating file list"):
        id_ = basename(config_dir)

        filelist[id_] = {
            'cam': join(config_dir, 'cam.json'),
            'cvis': join(config_dir, 'cvis.png'),
            'diffuse': join(config_dir, 'diffuse.png'),
            'light': join(config_dir, 'light.json'),
            'lvis': join(config_dir, 'lvis.png'),
            'nn': join(config_dir, 'nn.json'),
            'uv2cam': join(config_dir, 'uv2cam.npy')}

        if id_.startswith('trainvali_'):
            filelist[id_]['alpha'] = join(config_dir, 'alpha.png')
            filelist[id_]['rgb'] = join(config_dir, 'rgb.png')
            filelist[id_]['rgb_camspc'] = join(config_dir, 'rgb_camspc.png')

        # Check existence
        all_exist = True
        for _, v in filelist[id_].items():
            all_exist = all_exist and exists(v)
        filelist[id_]['complete'] = all_exist

        # Make the paths relative, to reduce the file size and make it
        # root-independent
        for k, v in filelist[id_].items():
            if k != 'complete':
                filelist[id_][k] = relpath(v, args.data_root)

    dump_json(filelist, args.out_json)


if __name__ == '__main__':
    main(parser.parse_args())
