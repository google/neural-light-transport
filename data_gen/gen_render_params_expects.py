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

"""An example script that generates task parameters (to be fed to `render.py`).
Each set of parameters is a light-camera configuration. These tasks can then be
distributed over many machines.

This file also generates, for each task, the expected output files. You may or
may not need this, but we are using them to figure out which tasks failed and
need re-running.

Example Usage:

    python "$ROOT"/neural-light-transport/data_gen/gen_render_params_expects.py \
        --mode='trainvali+test' \
        --scene="$ROOT"/data/scenes-v2/dragon_specular.blend \
        --cached_uv_unwrap="$ROOT"/data/scenes-v2/dragon_specular_uv.pickle \
        --trainvali_cams="$ROOT"'/data/trainvali_cams/*.json' \
        --test_cams="$ROOT"'/data/test_cams/*.json' \
        --trainvali_lights="$ROOT"'/data/trainvali_lights/*.json' \
        --test_lights="$ROOT"'/data/test_lights/*.json' \
        --cam_nn_json="$ROOT"/data/neighbors/cams.json \
        --light_nn_json="$ROOT"/data/neighbors/lights.json \
        --imh='512' \
        --uvs='1024' \
        --spp='256' \
        --outroot="$ROOT"/data/scenes-v2/dragon_specular_imh512_uvs1024_spp256/ \
        --jobdir="$ROOT"/tmp/specular
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import join, exists, abspath, basename
from glob import glob
from itertools import product


parser = ArgumentParser(description="")
parser.add_argument(
    '--mode', type=str, default='trainvali', help=(
        "generate data for which mode; allowed values: 'trainvali', 'test', "
        "'trainvali+test'"))
parser.add_argument(
    '--scene', type=str, required=True, help="path to the .blend scene")
parser.add_argument(
    '--cached_uv_unwrap', type=str, required=True,
    help="path to the .pickle UV unwrapping")
parser.add_argument(
    '--trainvali_cams', type=str, required=True, help=(
        "path to the camera .json used for training/validation; use a single "
        ".json if you don't want view variation (e.g., in relighting only)"))
parser.add_argument(
    '--test_cams', type=str, required=True, help=(
        "path to the camera .json used for testing; use a single .json if you "
        "don't want view variation (e.g., in relighting only)"))
parser.add_argument(
    '--cam_every', type=int, default=1, help="use one every N cameras")
parser.add_argument(
    '--trainvali_lights', type=str, required=True, help=(
        "path to the light .json used for training/validation; use a single "
        ".json if you don't want lighting variation (e.g., in view synthesis "
        "only)"))
parser.add_argument(
    '--test_lights', type=str, required=True, help=(
        "path to the light .json used for testing; use a single .json if you "
        "don't want lighting variation (e.g., in view synthesis only)"))
parser.add_argument(
    '--light_every', type=int, default=1, help="use one every N lights")
parser.add_argument(
    '--cam_nn_json', type=str, required=True,
    help="path to the .json of mapping from camera to its nearest neighbor")
parser.add_argument(
    '--light_nn_json', type=str, required=True,
    help="path to the .json of mapping from light to its nearest neighbor")
parser.add_argument(
    '--outroot', type=str, required=True, help="output directory root")
parser.add_argument(
    '--jobdir', type=str, required=True,
    help="directory holding task parameter files for worders to read")
parser.add_argument(
    '--imh', type=int, default=256,
    help="image height (width derived assuming the same aspect ratio)")
parser.add_argument(
    '--uvs', type=int, default=256, help="size of (square) texture map")
parser.add_argument(
    '--spp', type=int, default=64, help="samples per pixel for rendering")


def gen_tasks(args):
    # Glob cameras and lights
    trainvali_cams = sorted(glob(
        abspath(args.trainvali_cams)))[::args.cam_every]
    test_cams = sorted(glob(
        abspath(args.test_cams)))[::args.cam_every]
    trainvali_lights = sorted(glob(
        abspath(args.trainvali_lights)))[::args.light_every]
    test_lights = sorted(glob(
        abspath(args.test_lights)))[::args.light_every]

    # Generate training/validation tasks
    traivali_tasks = [
        (i, False, c, l) for i, (c, l) in enumerate(
            product(trainvali_cams, trainvali_lights))]

    # Generate testing tasks by casually pairing cameras and lights
    test_tasks = []
    for i, c in enumerate(test_cams):
        if i < len(test_lights):
            l = test_lights[i]
            test_tasks.append((i, True, c, l))

    if args.mode == 'trainvali':
        tasks = traivali_tasks
    elif args.mode == 'test':
        tasks = test_tasks
    elif args.mode == 'trainvali+test':
        tasks = traivali_tasks + test_tasks
    else:
        raise ValueError(args.mode)

    assert tasks, "No task generated"
    print(
        "Generating '%s' data: %d camera-light combinations" % (
            args.mode, len(tasks)))
    return tasks


def main(args):
    jobdir = abspath(args.jobdir)

    params_f = join(jobdir, 'render_params.txt')
    expects_f = join(jobdir, 'render_expects.txt')

    if not exists(jobdir):
        makedirs(jobdir)

    params_h = open(params_f, 'w')
    expects_h = open(expects_f, 'w')

    tasks = gen_tasks(args)

    for i, is_test, cam_json, light_json in tasks:
        pref = 'test' if is_test else 'trainvali'
        cam_name = basename(cam_json)[:-len('.json')]
        light_name = basename(light_json)[:-len('.json')]
        outdir = join(
            args.outroot, '{pref}_{i:09d}_{c}_{l}'.format(
                pref=pref, i=i, c=cam_name, l=light_name))

        params_h.write((
            '--scene={scene} --cached_uv_unwrap={unwrap} --cam_json={cam_json} '
            '--light_json={light_json} --cam_nn_json={cam_nn_json} '
            '--light_nn_json={light_nn_json} --imh={imh} --uvs={uvs} '
            '--spp={spp} --outdir={outdir}\n').format(
                scene=abspath(args.scene), unwrap=args.cached_uv_unwrap,
                cam_json=cam_json, light_json=light_json,
                cam_nn_json=args.cam_nn_json, light_nn_json=args.light_nn_json,
                imh=args.imh, uvs=args.uvs, spp=args.spp, outdir=outdir))
        expects_h.write(
            '{uv2cam} {cvis} {lvis} {rgb} {rgb_camspc}\n'.format(
                uv2cam=join(outdir, 'uv2cam.npy'),
                cvis=join(outdir, 'cvis.png'), lvis=join(outdir, 'lvis.png'),
                rgb=join(outdir, 'rgb.png'),
                rgb_camspc=join(outdir, 'rgb_camspc.png')))

    params_h.close()
    expects_h.close()

    print(
        "For task parameter files, see\n\t{param_f}\n\t{expect_f}".format(
            param_f=params_f, expect_f=expects_f))


if __name__ == '__main__':
    main(parser.parse_args())
