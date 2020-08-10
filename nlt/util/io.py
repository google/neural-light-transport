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

"""Lazy imports for XManager (which doesn't use the BUILD file).
"""

from os import makedirs, rename
from os.path import dirname, exists, isdir
from shutil import rmtree
from configparser import ConfigParser
import pickle as pkl
import json
import numpy as np

from . import logging as logutil


logger = logutil.Logger(loggee="util/io")


def restore(ckpt, ckptmanager):
    ckpt.restore(ckptmanager.latest_checkpoint)
    if ckptmanager.latest_checkpoint:
        logger.info("Resumed from step:\n\t%s", ckptmanager.latest_checkpoint)
    else:
        logger.info("Started from scratch")


def read_config(path):
    config = ConfigParser()
    with open(path, 'r') as h:
        config.read_file(h)
    return config


def prepare_outdir(outdir, overwrite=False, quiet=False):
    if isdir(outdir):
        # Directory already exists
        if not quiet:
            logger.info("Output directory already exisits:\n\t%s", outdir)
        if overwrite:
            rmtree(outdir)
            if not quiet:
                logger.warn("Output directory wiped:\n\t%s", outdir)
        else:
            if not quiet:
                logger.info("Overwrite is off, so doing nothing")
            return
    makedirs(outdir)


def write_pickle(data, outpath):
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir)

    with open(outpath, 'wb') as h:
        pkl.dump(data, h)


def read_pickle(path):
    with open(path, 'rb') as h:
        data = pkl.load(h)
    return data


def imwrite_tensor(tensor_uint, out_prefix):
    from PIL import Image

    for i in range(tensor_uint.shape[0]):
        out_path = out_prefix + '_%03d.png' % i
        makedirs(dirname(out_path))
        arr = tensor_uint[i, :, :, :].numpy()
        img = Image.fromarray(arr)
        with open(out_path, 'wb') as h:
            img.save(h)


def write_video(imgs, outpath, fps=12):
    import xiuminglib as xm

    assert imgs, "No image"
    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir)

    # Make sure 3-channel
    frames = []
    for img in imgs:
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        frames.append(img)

    xm.vis.video.make_video(frames, fps=fps, outpath=outpath)


def write_apng(imgs, labels, outprefix):
    import xiuminglib as xm

    apng_f = outprefix + '.apng'
    xm.vis.video.make_apng(
        imgs, labels=labels, label_top_left_xy=(200, 200), font_size=100,
        font_color=(1, 1, 1), outpath=apng_f)
    # Renaming for easier insertion into slides
    png_f = apng_f.replace('.apng', '.png')
    rename(apng_f, png_f, overwrite=True)
    return png_f


def read_json(path):
    with open(path, 'r') as h:
        data = json.load(h)
    return data


def write_json(data, path):
    out_dir = dirname(path)
    if not exists(out_dir):
        makedirs(out_dir)

    with open(path, 'w') as h:
        json.dump(data, h, indent=4, sort_keys=True)


def read_npy(path):
    with open(path, 'rb') as h:
        data = np.load(h)
    return data
