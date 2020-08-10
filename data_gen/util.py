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

from copy import deepcopy
from os import makedirs
from os.path import exists, dirname, basename
import json
import numpy as np
import cv2


def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data


def dump_json(data, path):
    """Pretty dump.
    """
    dir_ = dirname(path)
    if not exists(dir_):
        makedirs(dir_)

    with open(path, 'w') as h:
        json.dump(data, h, indent=4, sort_keys=True)


def safe_cast_to_int(float_):
    assert float_ == int(float_), "Failed to safely cast %f to integer" % float_
    return int(float_)


def remap(src, mapping, force_kbg=True):
    h, w = src.shape[:2]
    mapping_x = mapping[:, :, 0] * w
    mapping_y = mapping[:, :, 1] * h
    mapping_x = mapping_x.astype(np.float32)
    mapping_y = mapping_y.astype(np.float32)

    src_ = deepcopy(src)
    if force_kbg:
        # Set left-top corner (where background takes colors from) to black
        src_[0, 0, ...] = 0

    dst = cv2.remap(src_, mapping_x, mapping_y, cv2.INTER_LINEAR)
    return dst


def add_b_ch(img_rg):
    assert img_rg.ndim == 3 and img_rg.shape[2] == 2, "Input should be HxWx2"
    img_rgb = np.dstack((img_rg, np.zeros_like(img_rg)[:, :, :1]))
    return img_rgb


def save_float16_npy(data, path):
    """Use float16 for faster IO during training.
    """
    np.save(path, data.astype(np.float16))


def name_from_json_path(json_path):
    return basename(json_path)[:-len('.json')]
