from sys import argv
from os.path import join
from argparse import ArgumentParser
from glob import glob
import numpy as np

import bpy

import xiuminglib as xm

from util import load_json, dump_json


parser = ArgumentParser(description="")
parser.add_argument(
    '--scene', type=str, required=True, help="path to the .blend scene")
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


def main(args):
    # Open scene
    xm.blender.scene.open_blend(args.scene)

    # Remove existing cameras and lights, if any
    for o in bpy.data.objects:
        o.select = o.type in ('LAMP', 'CAMERA')
    bpy.ops.object.delete()

    trainvali_cams = sorted(glob(args.trainvali_cams))
    trainvali_lights = sorted(glob(args.trainvali_lights))
    test_cams = sorted(glob(args.test_cams))
    test_lights = sorted(glob(args.test_lights))

    bpy.context.user_preferences.system.use_scripts_auto_execute = True
    scene = bpy.context.scene
    scene.frame_start = 1

    # Training/validation cameras and lights: sequential; cameras first
    n_frames = len(trainvali_cams + trainvali_lights)
    scene.frame_end = n_frames
    for i, cam_json in enumerate(trainvali_cams):
        cam = load_json(cam_json)
        cam_obj = xm.blender.camera.add_camera(
            xyz=cam['position'], rot_vec_rad=cam['rotation'],
            f=cam['focal_length'], sensor_width=cam['sensor_width'],
            sensor_height=cam['sensor_height'])
        # Make it smaller for just visualization
        cam_obj.scale = (0.1, 0.1, 0.1)
        # Add visibility driver
        visible_frames = list(range(i + 1, n_frames + 1))
        make_visible_in(cam_obj, visible_frames)
    for j, light_json in enumerate(trainvali_lights):
        light = load_json(light_json)
        light_obj = xm.blender.light.add_light_point(light['position'])
        # Add visibility driver
        visible_frames = list(range(j + len(trainvali_cams), n_frames + 1))
        make_visible_in(light_obj, visible_frames)
    # Save this visualization scene
    outf = join(args.outdir, 'trainvali.blend')
    xm.blender.scene.save_blend(outf)

    # Remove training/validation cameras and lights
    for o in bpy.data.objects:
        o.select = o.type in ('LAMP', 'CAMERA')
    bpy.ops.object.delete()

    # Test cameras and lights: cameras and lights in sync.
    n_frames = len(test_cams)
    assert n_frames == len(test_lights), "Cameras and lights must be 1:1 paired"
    scene.frame_end = n_frames
    for i, (cam_json, light_json) in enumerate(zip(test_cams, test_lights)):
        cam = load_json(cam_json)
        light = load_json(light_json)
        cam_obj = xm.blender.camera.add_camera(
            xyz=cam['position'], rot_vec_rad=cam['rotation'],
            f=cam['focal_length'], sensor_width=cam['sensor_width'],
            sensor_height=cam['sensor_height'])
        light_obj = xm.blender.light.add_light_point(light['position'])
        # Make it smaller for just visualization
        cam_obj.scale = (0.1, 0.1, 0.1)
        # Add visibility driver
        visible_frames = list(range(i + 1, n_frames + 1))
        make_visible_in(cam_obj, visible_frames)
        make_visible_in(light_obj, visible_frames)

    # Save this visualization scene
    outf = join(args.outdir, 'test.blend')
    xm.blender.scene.save_blend(outf)


def make_visible_in(obj, visible_frames):
    obj['vis_in'] = visible_frames # custom property

    # Add a driver for view visibility
    driver = obj.driver_add('hide').driver
    var = driver.variables.new()
    var.name = 'vis_in'
    var.targets[0].id = obj
    var.targets[0].data_path = '["vis_in"]'
    logical_expr = 'bpy.data.scenes["Scene"].frame_current not in vis_in'
    driver.expression = logical_expr

    # Add a driver for render visibility
    driver = obj.driver_add('hide_render').driver
    var = driver.variables.new()
    var.name = 'vis_in'
    var.targets[0].id = obj
    var.targets[0].data_path = '["vis_in"]'
    driver.expression = logical_expr


if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]

    main(parser.parse_args(argv))
