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

"""For portability and easy deployment, this script renders a single
light-camera configuration. Depending on your computing environment, you can
dispatch this script to multiple machines to render all light-camera
configurations in parallel.

Note that there is still some parallelism when rendering a single configuration,
because Blender automatically renders using multiple threads: all CPUs will be
used for rendering.

Example Usage:

PYTHONPATH="$ROOT"/neural-light-transport/data_gen:"$PYTHONPATH" \
    "$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
        --background \
        --python "$ROOT"/neural-light-transport/data_gen/render.py \
        -- \
        --scene="$ROOT"/data/scenes-v2/dragon_specular.blend \
        --cached_uv_unwrap="$ROOT"/data/scenes-v2/dragon_specular_uv.pickle \
        --cam_json="$ROOT"/data/trainvali_cams/P28R.json \
        --light_json="$ROOT"/data/trainvali_lights/L330.json \
        --cam_nn_json="$ROOT"/data/neighbors/cams.json \
        --light_nn_json="$ROOT"/data/neighbors/lights.json \
        --imh='512' \
        --uvs='1024' \
        --spp='256' \
        --outdir="$ROOT"/output/render/dragon/trainvali_000021183_P28R_L330
"""

from sys import argv
from argparse import ArgumentParser
from shutil import copyfile
from os.path import join
import pickle as pk
import numpy as np
from tqdm import tqdm

# Blender
import bpy
from mathutils import Vector, bvhtree

import xiuminglib as xm

from util import load_json, dump_json, safe_cast_to_int, remap, add_b_ch, \
    save_float16_npy, name_from_json_path


parser = ArgumentParser(description="")
parser.add_argument(
    '--scene', type=str, required=True, help="path to the .blend scene")
parser.add_argument(
    '--cached_uv_unwrap', type=str, required=True, help=(
        "path to the cached .pickle of UV unwrapping, which needs doing only "
        "once per scene"))
parser.add_argument(
    '--cam_json', type=str, required=True, help="path to the camera .json")
parser.add_argument(
    '--light_json', type=str, required=True, help="path to the light .json")
parser.add_argument(
    '--cam_nn_json', type=str, required=True,
    help="path to the .json of mapping from camera to its nearest neighbor")
parser.add_argument(
    '--light_nn_json', type=str, required=True,
    help="path to the .json of mapping from light to its nearest neighbor")
parser.add_argument(
    '--imh', type=int, default=256,
    help="image height (width derived from camera's sensor height and width)")
parser.add_argument(
    '--uvs', type=int, default=256, help="size of (square) texture map")
parser.add_argument(
    '--spp', type=int, default=64, help="samples per pixel for rendering")
parser.add_argument(
    '--outdir', type=str, required=True, help="output directory")
parser.add_argument(
    '--debug', type=bool, default=False,
    help="whether to dump additional outputs for debugging")


def main(args):
    # Open scene
    xm.blender.scene.open_blend(args.scene)
    obj = bpy.data.objects['object']

    # Remove existing cameras and lights, if any
    for o in bpy.data.objects:
        o.select = o.type in ('LAMP', 'CAMERA')
    bpy.ops.object.delete()

    # Load camera and light
    cam = load_json(args.cam_json)
    light = load_json(args.light_json)

    # Add camera and light
    cam_obj = xm.blender.camera.add_camera(
        xyz=cam['position'], rot_vec_rad=cam['rotation'],
        name=cam['name'], f=cam['focal_length'],
        sensor_width=cam['sensor_width'], sensor_height=cam['sensor_height'],
        clip_start=cam['clip_start'], clip_end=cam['clip_end'])
    xm.blender.light.add_light_point(
        xyz=light['position'], name=light['name'], size=light['size'])

    # Common rendering settings
    xm.blender.render.easyset(n_samples=args.spp, color_mode='RGB')

    # Image and texture resolution
    imw = args.imh / cam['sensor_height'] * cam['sensor_width']
    imw = safe_cast_to_int(imw)
    xm.blender.render.easyset(h=args.imh, w=imw)

    # Render full RGB
    # TODO: Render in .exr to avoid saturated pixels (and tone mapping)
    rgb_camspc_f = join(args.outdir, 'rgb_camspc.png')
    xm.blender.render.render(rgb_camspc_f)
    rgb_camspc = xm.io.img.load(rgb_camspc_f, as_array=True)[:, :, :3]

    # Render alpha
    alpha_f = join(args.outdir, 'alpha.png')
    xm.blender.render.render_alpha(alpha_f, samples=args.spp)
    alpha = xm.io.img.load(alpha_f, as_array=True)
    alpha = xm.img.normalize_uint(alpha)

    # Cast rays through all pixels to the object
    xs, ys = np.meshgrid(range(imw), range(args.imh))
    # (0, 0)
    # +--------> (w, 0)
    # |           x
    # |
    # v y (0, h)
    xys = np.dstack((xs, ys)).reshape(-1, 2)
    ray_tos, x_locs, x_objnames, x_facei, x_normals = \
        xm.blender.camera.backproject_to_3d(
            xys, cam_obj, obj_names=obj.name, world_coords=True)
    intersect = {
        'ray_tos': ray_tos, 'obj_names': x_objnames, 'face_i': x_facei,
        'locs': x_locs, 'normals': x_normals}

    # Compute mapping between UV and camera space
    uv2cam, cam2uv = calc_bidir_mapping(
        args.cached_uv_unwrap, obj.name, xys, intersect, args.uvs)
    uv2cam = add_b_ch(uv2cam)
    cam2uv = add_b_ch(cam2uv)
    uv2cam[alpha < 1] = 0 # mask out interpolated values that fall outside
    xm.io.img.write_arr(uv2cam, join(args.outdir, 'uv2cam.png'), clip=True)
    xm.io.img.write_arr(cam2uv, join(args.outdir, 'cam2uv.png'), clip=True)
    save_float16_npy(uv2cam[:, :, :2], join(args.outdir, 'uv2cam.npy'))
    save_float16_npy(cam2uv[:, :, :2], join(args.outdir, 'cam2uv.npy'))

    # Compute light cosines, considering occlusion
    lvis_camspc = calc_light_cosines(
        light['position'], xys, intersect, obj)
    lvis_camspc = xm.img.denormalize_float(np.clip(lvis_camspc, 0, 1))
    xm.io.img.write_img(lvis_camspc, join(args.outdir, 'lvis_camspc.png'))

    # Compute view cosines
    cvis_camspc = calc_view_cosines(
        cam_obj.location, xys, intersect, obj.name)
    cvis_camspc = xm.img.denormalize_float(np.clip(cvis_camspc, 0, 1))
    xm.io.img.write_img(cvis_camspc, join(args.outdir, 'cvis_camspc.png'))

    # Remap buffers to UV space
    cvis = remap(cvis_camspc, cam2uv)
    lvis = remap(lvis_camspc, cam2uv)
    rgb = remap(rgb_camspc, cam2uv)
    xm.io.img.write_img(cvis, join(args.outdir, 'cvis.png'))
    xm.io.img.write_img(lvis, join(args.outdir, 'lvis.png'))
    xm.io.img.write_img(rgb, join(args.outdir, 'rgb.png'))
    if args.debug:
        # Remap it backwards to check if we get back the camera-space buffer
        # TODO: UV wrapped images may have seams/holes due to interpolation
        # errors (fixable by better engineering), but this should be fine
        # because the network will learn to eliminate such artifacts in
        # trying to match the camera-space ground truth
        cvis_camspc_repro = remap(cvis, uv2cam)
        lvis_camspc_repro = remap(lvis, uv2cam)
        rgb_camspc_repro = remap(rgb, uv2cam)
        xm.io.img.write_img(
            cvis_camspc_repro, join(args.outdir, 'cvis_camspc_repro.png'))
        xm.io.img.write_img(
            lvis_camspc_repro, join(args.outdir, 'lvis_camspc_repro.png'))
        xm.io.img.write_img(
            rgb_camspc_repro, join(args.outdir, 'rgb_camspc_repro.png'))

    # Dump camera and light
    copyfile(args.cam_json, join(args.outdir, 'cam.json'))
    copyfile(args.light_json, join(args.outdir, 'light.json'))

    # Dump neighbor information
    cam_nn = load_json(args.cam_nn_json)
    light_nn = load_json(args.light_nn_json)
    cam_name = name_from_json_path(args.cam_json)
    light_name = name_from_json_path(args.light_json)
    nn = {'cam': cam_nn[cam_name], 'light': light_nn[light_name]}
    dump_json(nn, join(args.outdir, 'nn.json'))


def calc_view_cosines(cam_loc, xys, intersect, obj_name):
    imw = xys[:, 0].max() + 1
    imh = xys[:, 1].max() + 1

    view_cosines = np.zeros((imh, imw))

    for oname, xy, loc, normal in tqdm(
            zip(
                intersect['obj_names'], xys, intersect['locs'],
                intersect['normals']),
            total=xys.shape[0], desc="Filling view cosines"):
        if loc is None or oname != obj_name:
            continue

        p2c = (cam_loc - loc).normalized()
        normal = normal.normalized()

        view_cosines[xy[1], xy[0]] = p2c.dot(normal)

    return view_cosines


def calc_light_cosines(light_loc, xys, cam_intersect, obj):
    """Self-occlusion is considered here, so pixels in cast shadow have 0
    cosine values.
    """
    light_loc = Vector(light_loc)

    # Cast rays from the light to determine occlusion
    bm = xm.blender.object.get_bmesh(obj)
    tree = bvhtree.BVHTree.FromBMesh(bm)
    world2obj = obj.matrix_world.inverted()
    occluded = [False] * len(cam_intersect['locs'])
    for i, loc in enumerate(cam_intersect['locs']):
        if loc is not None:
            ray_from = world2obj * light_loc
            ray_to = world2obj * loc
            _, _, _, ray_dist = xm.blender.object.raycast(
                tree, ray_from, ray_to)
            if ray_dist is None:
                # Numerical issue, but hey, the ray is not blocked
                occluded[i] = False
            else:
                reach = np.isclose(ray_dist, (ray_to - ray_from).magnitude)
                occluded[i] = not reach

    imw = xys[:, 0].max() + 1
    imh = xys[:, 1].max() + 1

    light_cosines = np.zeros((imh, imw))

    for oname, xy, loc, normal, occlu in tqdm(
            zip(
                cam_intersect['obj_names'], xys, cam_intersect['locs'],
                cam_intersect['normals'], occluded),
            total=xys.shape[0], desc="Filling light cosines"):
        if loc is None or oname != obj.name:
            continue

        if occlu:
            continue

        p2l = (Vector(light_loc) - loc).normalized()
        normal = normal.normalized()

        light_cosines[xy[1], xy[0]] = p2l.dot(normal)

    return light_cosines


def calc_bidir_mapping(
        cached_unwrap, obj_name, xys, intersect, uvs, max_l1_interp=4):
    imw = xys[:, 0].max() + 1
    imh = xys[:, 1].max() + 1

    # Load the UV unwrapping by Blender
    with open(cached_unwrap, 'rb') as h:
        fi_li_vi_u_v = pk.load(h)

    # UV convention:
    # (0, 1)
    #   ^ v
    #   |
    #   |
    #   +------> (1, 0)
    # (0, 0)   u

    # Collect locations and their associated values
    uv2cam_locs, uv2cam_vals = [], []
    cam2uv_locs, cam2uv_vals = [], []
    for xy, oname, fi in tqdm(
            zip(xys, intersect['obj_names'], intersect['face_i']),
            total=xys.shape[0], desc="Filling camera-UV mappings"):
        if fi is None or oname != obj_name:
            continue

        uv = fi_li_vi_u_v[fi][:, 2:]

        # Collect locations and values for UV to camera
        camspc_loc = (xy[0] / float(imw), 1 - xy[1] / float(imh))
        uvspc_loc = np.hstack((uv[:, :1], 1 - uv[:, 1:]))
        uv2cam_locs.append(np.vstack([camspc_loc] * uvspc_loc.shape[0]))
        uv2cam_vals.append(uvspc_loc)

        # Now for camera to UV
        uvspc_loc = uv
        camspc_loc = (xy[0] / float(imw), xy[1] / float(imh))
        cam2uv_locs.append(uvspc_loc)
        cam2uv_vals.append(np.vstack([camspc_loc] * uvspc_loc.shape[0]))

    # Location convention for xm.img.grid_query_unstruct():
    # (0, 1)
    #     ^ v
    #     |
    #     +------> (1, 0)
    # (0, 0)      u

    # Value convention for xm.img.grid_query_unstruct(), for use by remap():
    # (0, 0)
    # +--------> (1, 0)
    # |           x
    # |
    # v y (0, 1)

    interp_method = {
        'func': 'griddata',
        'func_underlying': 'nearest',
        'fill_value': (0,), # black
        'max_l1_interp': max_l1_interp}

    # UV to camera space: interpolate unstructured values into an image
    locs = np.vstack(uv2cam_locs)
    vals = np.vstack(uv2cam_vals)
    uv2cam = xm.img.grid_query_unstruct(
        locs, vals, (imh, imw), method=interp_method)

    # Camera to UV space: interpolate unstructured values into an image
    locs = np.vstack(cam2uv_locs)
    vals = np.vstack(cam2uv_vals)
    cam2uv = xm.img.grid_query_unstruct(
        locs, vals, (uvs, uvs), method=interp_method)

    return uv2cam, cam2uv


if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]

    main(parser.parse_args(argv))
