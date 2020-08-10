from sys import argv
from shutil import rmtree
from argparse import ArgumentParser
from glob import glob
from os.path import join, exists
import numpy as np

import xiuminglib as xm

from util import load_json, dump_json


parser = ArgumentParser(description="")
parser.add_argument(
    '--scene', type=str, required=True, help="path to the .blend scene")
parser.add_argument(
    '--trainvali_cams', type=str, required=True,
    help="path to the camera .json used for training/validation")
parser.add_argument(
    '--trainvali_lights', type=str, required=True,
    help="path to the light .json used for training/validation")
parser.add_argument(
    '--n_360', type=int, default=100, help="number of points along 360")
parser.add_argument(
    '--n_sway', type=int, default=25, help="number of points along each sway")
parser.add_argument(
    '--cam_outdir', type=str, required=True, help="camera output directory")
parser.add_argument(
    '--light_outdir', type=str, required=True, help="light output directory")


def main(args):
    if exists(args.cam_outdir):
        rmtree(args.cam_outdir)
    if exists(args.light_outdir):
        rmtree(args.light_outdir)

    # Open scene
    xm.blender.scene.open_blend(args.scene)

    # Figure out physical cameras' and lights' radii
    cams = []
    for cam_json in sorted(glob(args.trainvali_cams)):
        cam = load_json(cam_json)
        cams.append(cam)
    cam_mean_r = compute_avg_r(cams)
    any_cam = cams[0]
    lights = []
    for light_json in sorted(glob(args.trainvali_lights)):
        light = load_json(light_json)
        lights.append(light)
    light_mean_r = compute_avg_r(lights)
    any_light = lights[0]

    global_cam_i, global_light_i = 0, 0

    # First cameras and lights do a 360 together
    cam_rlatlng, light_rlatlng = [], []
    lat = 0
    for lng in np.linspace(0, 2 * np.pi, args.n_360):
        cam_rlatlng.append((cam_mean_r, lat, lng))
        light_rlatlng.append((light_mean_r, lat, lng))
    cam_rlatlng = np.array(cam_rlatlng)
    light_rlatlng = np.array(light_rlatlng)
    cam_xyz = xm.geometry.sph.sph2cart(cam_rlatlng)
    light_xyz = xm.geometry.sph.sph2cart(light_rlatlng)
    # Dump
    for cam_loc, light_loc in zip(cam_xyz, light_xyz):
        # Cameras
        cam_obj = xm.blender.camera.add_camera(
            xyz=cam_loc, sensor_width=any_cam['sensor_width'],
            sensor_height=any_cam['sensor_height'])
        xm.blender.camera.point_camera_to(cam_obj, (0, 0, 0))
        any_cam['name'] = 'c%03d' % global_cam_i
        any_cam['rotation'] = tuple(cam_obj.rotation_euler)
        any_cam['position'] = tuple(cam_obj.location)
        cam_json = join(args.cam_outdir, 'c%03d.json' % global_cam_i)
        dump_json(any_cam, cam_json)
        global_cam_i += 1
        # Lights
        light_json = join(args.light_outdir, 'l%03d.json' % global_light_i)
        any_light['name'] = 'l%03d' % global_light_i
        any_light['position'] = tuple(light_loc)
        dump_json(any_light, light_json)
        global_light_i += 1

    # Cameras move in a cross
    cam_rlatlng = []
    for lng in np.hstack((
            np.linspace(0, -np.pi / 4, args.n_sway),
            np.linspace(-np.pi / 4, 0, args.n_sway),
            np.linspace(0, np.pi / 4, args.n_sway),
            np.linspace(np.pi / 4, 0, args.n_sway))):
        cam_rlatlng.append((cam_mean_r, 0, lng))
    for lat in np.hstack((
            np.linspace(0, -np.pi / 4, args.n_sway),
            np.linspace(-np.pi / 4, 0, args.n_sway),
            np.linspace(0, np.pi / 4, args.n_sway),
            np.linspace(np.pi / 4, 0, args.n_sway))):
        cam_rlatlng.append((cam_mean_r, lat, 0))
    cam_rlatlng = np.array(cam_rlatlng)
    cam_xyz = xm.geometry.sph.sph2cart(cam_rlatlng)
    sampled_rlatlng = xm.geometry.sph.uniform_sample_sph(
        2 * len(cam_rlatlng), r=light_mean_r, convention='lat-lng')
    light_xyz = xm.geometry.sph.sph2cart(sampled_rlatlng)
    light_xyz = light_xyz[(light_xyz.shape[0] // 2):] # front only
    coord_trans = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    light_xyz = coord_trans.dot(light_xyz.T).T
    assert light_xyz.shape[0] == cam_xyz.shape[0], \
        "Cameras and lights must be paired 1:1"
    # Dump
    for cam_loc, light_loc in zip(cam_xyz, light_xyz):
        # Cameras
        cam_obj = xm.blender.camera.add_camera(
            xyz=cam_loc, sensor_width=any_cam['sensor_width'],
            sensor_height=any_cam['sensor_height'])
        xm.blender.camera.point_camera_to(cam_obj, (0, 0, 0))
        any_cam['name'] = 'c%03d' % global_cam_i
        any_cam['rotation'] = tuple(cam_obj.rotation_euler)
        any_cam['position'] = tuple(cam_obj.location)
        cam_json = join(args.cam_outdir, 'c%03d.json' % global_cam_i)
        dump_json(any_cam, cam_json)
        global_cam_i += 1
        # Lights
        light_json = join(args.light_outdir, 'l%03d.json' % global_light_i)
        any_light['name'] = 'l%03d' % global_light_i
        any_light['position'] = tuple(light_loc)
        dump_json(any_light, light_json)
        global_light_i += 1


def compute_avg_r(objs):
    xyz = []
    for obj in objs:
        xyz.append(obj['position'])
    xyz = np.vstack(xyz)
    rlatlng = xm.geometry.sph.cart2sph(xyz)
    mean_r = np.mean(rlatlng[:, 0])
    return mean_r


if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]

    main(parser.parse_args(argv))
