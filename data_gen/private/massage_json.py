"""Run this ONLY ONCE on the JSON files copied from CNS.
"""
from sys import argv
from argparse import ArgumentParser
from glob import glob
import numpy as np

import xiuminglib as xm

from util import load_json, dump_json


parser = ArgumentParser(description="")
parser.add_argument(
    '--scene', type=str, required=True, help="path to the .blend scene")
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


def main(args):
    # Open scene
    xm.blender.scene.open_blend(args.scene)

    coord_trans = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Cameras
    for cam_json in sorted(glob(args.trainvali_cams)) + \
            sorted(glob(args.test_cams)):
        cam = load_json(cam_json)
        cam.pop('sensor_weight')
        cam['sensor_width'] = 16
        cam['sensor_height'] = 16
        cam['focal_length'] = 18

        cpos = coord_trans.dot(cam['position'])
        cpos -= np.array([0, 0, 1.575])
        # Origin is stage center now

        # Make look at origin
        cam_obj = xm.blender.camera.add_camera(
            xyz=cpos, f=cam['focal_length'],
            sensor_width=cam['sensor_width'],
            sensor_height=cam['sensor_height'])
        xm.blender.camera.point_camera_to(cam_obj, (0, 0, 0))

        # Override locaiton and rotation
        cam['rotation'] = tuple(cam_obj.rotation_euler)
        cam['position'] = tuple(cam_obj.location)

        # Dump it back
        dump_json(cam, cam_json)

    # Lights
    for light_json in sorted(glob(args.trainvali_lights)) + \
            sorted(glob(args.test_lights)):
        light = load_json(light_json)

        lpos = coord_trans.dot(light['position'])
        lpos -= np.array([0, 0, 1.575])
        # Origin is stage center now

        light['position'] = tuple(lpos)
        light['size'] = 1

        dump_json(light, light_json)


if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]

    main(parser.parse_args(argv))
