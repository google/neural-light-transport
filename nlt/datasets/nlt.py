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

# pylint: disable=relative-beyond-top-level

from os.path import join, exists
from itertools import product
import re
import numpy as np
from PIL import Image

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import xiuminglib as xm

from util import logging as logutil, io as ioutil
from .base import Dataset as BaseDataset


logger = logutil.Logger(loggee="datasets/nlt")


class Dataset(BaseDataset):
    def __init__(self, config, mode, **kwargs):
        self.data_root = config.get('DEFAULT', 'data_root')
        data_status_path = self.data_root.rstrip('/') + '.json'
        if not exists(data_status_path):
            raise FileNotFoundError((
                "Data status JSON not found at \n\t%s\nRun "
                "$REPO/data_gen/postproc.py to generate it") % data_status_path)
        self.data_paths = ioutil.read_json(data_status_path)
        # Because paths in JSON are relative, prepend data root directory
        for _, paths in self.data_paths.items():
            for k, v in paths.items():
                if k != 'complete':
                    paths[k] = join(self.data_root, v)
        super().__init__(config, mode, **kwargs)
        # Trigger init. in a main thread before starting multi-threaded work.
        # See http://yaqs/eng/q/6292200559345664 for details
        Image.init()

    def _glob(self):
        # Handle holdouts
        holdout_cam = self.config.get('DEFAULT', 'holdout_cam').split(',')
        holdout_light = self.config.get('DEFAULT', 'holdout_light').split(',')
        holdout = [
            '%s_%s' % x for x in product(holdout_cam, holdout_light)]
        # Add only if data are complete for this camera
        ids = []
        for id_, paths in self.data_paths.items():
            if id_.startswith('test' if self.mode == 'test' else 'trainvali'):
                if paths['complete']:
                    ids.append(id_)
                else:
                    logger.warn(
                        "Skipping '%s' because its data are incomplete", id_)
        # Shortcircuit if testing
        if self.mode == 'test':
            logger.info(
                "Number of '%s' camera-light combinations: %d", self.mode,
                len(ids))
            return ids
        # Training-validation split
        ids_split = []
        for id_ in ids:
            # ID is {bin_mode}_{i:09d}_{cam}_{light}
            cam_light = '_'.join(id_.split('_')[-2:])
            if (self.mode == 'vali' and cam_light in holdout) or \
                    (self.mode != 'vali' and cam_light not in holdout):
                ids_split.append(id_)
        logger.info(
            "Number of '%s' camera-light combinations: %d", self.mode,
            len(ids_split))
        return ids_split

    def _get_nn_id(self, nn):
        id_regex = re.compile(
            r'trainvali_\d\d\d\d\d\d\d\d\d_{cam}_{light}'.format(**nn))
        matched = [
            x for x in self.data_paths.keys() if id_regex.search(x) is not None]
        n_matches = len(matched)
        if not matched:
            return None
        if n_matches == 1:
            return matched[0]
        raise ValueError(
            "Found {n} matches:\n\t{matches}".format(
                n=n_matches, matches=matched))

    def _process_example_precache(self, id_): # pylint: disable=arguments-differ
        """Loads data from paths.
        """
        id_, base, cvis, lvis, warp, rgb, rgb_camspc, nn_id, nn_base, nn_rgb, \
            nn_rgb_camspc = tf.py_function(
                self._load_data, [id_], (
                    tf.string, tf.float32, tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.string, tf.float32, tf.float32,
                    tf.float32))
        return \
            id_, base, cvis, lvis, warp, rgb, rgb_camspc, nn_id, nn_base, \
            nn_rgb, nn_rgb_camspc

    def _load_data(self, id_):
        if isinstance(id_, tf.Tensor):
            id_ = id_.numpy().decode()
        paths = self.data_paths[id_]
        imh = self.config.getint('DEFAULT', 'imh')
        imw = self.config.getint('DEFAULT', 'imw')
        # Load images
        base = xm.io.img.load(paths['diffuse'], as_array=True)[:, :, :3]
        cvis = xm.io.img.load(paths['cvis'], as_array=True)
        lvis = xm.io.img.load(paths['lvis'], as_array=True)
        warp = ioutil.read_npy(paths['uv2cam'])
        if self.mode == 'test':
            rgb = np.zeros_like(base) # placeholders
            rgb_camspc = np.zeros((imh, imw, 3))
        else:
            rgb = xm.io.img.load(paths['rgb'], as_array=True)[:, :, :3]
            rgb_camspc = xm.io.img.load(
                paths['rgb_camspc'], as_array=True)[:, :, :3]
        # Normalize to [0, 1]
        base = xm.img.normalize_uint(base)
        cvis = xm.img.normalize_uint(cvis)
        lvis = xm.img.normalize_uint(lvis)
        if self.mode != 'test':
            rgb = xm.img.normalize_uint(rgb)
            rgb_camspc = xm.img.normalize_uint(rgb_camspc)
        # Resize images
        uvh = self.config.getint('DEFAULT', 'uvh')
        base = xm.img.resize(base, new_h=uvh)
        cvis = xm.img.resize(cvis, new_h=uvh)
        lvis = xm.img.resize(lvis, new_h=uvh)
        rgb = xm.img.resize(rgb, new_h=uvh)
        rgb_camspc = xm.img.resize(rgb_camspc, new_h=imh, new_w=imw)
        # NOTE: We didn't resize warp because this introduces artifacts --
        # always warp first and then resize
        # Neighbor diffuse base and full
        nn = ioutil.read_json(paths['nn'])
        nn_id = self._get_nn_id(nn)
        if nn_id is None:
            nn_id = 'incomplete-data_{cam}_{light}'.format(**nn)
            # NOTE: When neighbor is missing, simply return black placeholders
            nn_base = np.zeros_like(base)
            nn_rgb = np.zeros_like(rgb)
            nn_rgb_camspc = np.zeros_like(rgb_camspc)
        else:
            nn_base = xm.io.img.load(
                self.data_paths[nn_id]['diffuse'], as_array=True)[:, :, :3]
            nn_rgb = xm.io.img.load(
                self.data_paths[nn_id]['rgb'], as_array=True)[:, :, :3]
            nn_rgb_camspc = xm.io.img.load(
                self.data_paths[nn_id]['rgb_camspc'], as_array=True)[:, :, :3]
            nn_rgb_camspc = nn_rgb_camspc[:, :, :3] # discards alpha
            nn_base = xm.img.normalize_uint(nn_base)
            nn_rgb = xm.img.normalize_uint(nn_rgb)
            nn_rgb_camspc = xm.img.normalize_uint(nn_rgb_camspc)
            nn_base = xm.img.resize(nn_base, new_h=uvh)
            nn_rgb = xm.img.resize(nn_rgb, new_h=uvh)
            nn_rgb_camspc = xm.img.resize(nn_rgb_camspc, new_h=imh, new_w=imw)
        # Return
        base = base.astype(np.float32)
        cvis = cvis.astype(np.float32)[:, :, None] # HxWx1
        lvis = lvis.astype(np.float32)[:, :, None]
        warp = warp.astype(np.float32)
        rgb = rgb.astype(np.float32)
        rgb_camspc = rgb_camspc.astype(np.float32)
        nn_base = nn_base.astype(np.float32)
        nn_rgb = nn_rgb.astype(np.float32)
        nn_rgb_camspc = nn_rgb_camspc.astype(np.float32)
        return \
            id_, base, cvis, lvis, warp, rgb, rgb_camspc, nn_id, nn_base, \
            nn_rgb, nn_rgb_camspc
