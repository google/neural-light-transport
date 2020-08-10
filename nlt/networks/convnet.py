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

import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from util import logging as logutil, net as netutil
from .seq import Network as BaseNetwork
from .elements import conv, norm, act, pool, iden, deconv, upconv


logger = logutil.Logger(loggee="networks/convnet")


class Network(BaseNetwork):
    def __init__(
            self, depth0, depth, kernel, stride, norm_type=None,
            act_type='relu', pool_type=None):
        super().__init__()
        norm_type = self.str2none(norm_type)
        pool_type = self.str2none(pool_type)
        min_n_ch = depth0
        max_n_ch = depth
        n_feat = netutil.gen_feat_n(min_n_ch, max_n_ch)
        # Stacking the layers
        prev_n = 0
        self.is_contracting, self.spatsize_changes = [], []
        # 1x1 conv to generate an original-res. feature map
        self.layers.append(conv(1, n_feat[0], stride=1))
        self.is_contracting.append(True)
        self.spatsize_changes.append(1)
        for n in n_feat[:-1]:
            # Contracting spatially
            if n >= prev_n: # so 64 -> 64 is considered "contracting"
                self.layers.append(
                    tf.keras.Sequential([
                        conv(kernel, n, stride=stride),
                        norm(norm_type), # could be identity
                        act(act_type),
                        conv(kernel, n, stride=1),
                        norm(norm_type),
                        act(act_type),
                        pool(pool_type), # could be identity
                    ]))
                self.is_contracting.append(True)
                spatsize_change = 1 / stride
                if pool_type is not None:
                    spatsize_change *= 1 / 2
                self.spatsize_changes.append(spatsize_change)
            # Expanding spatially
            else:
                self.layers.append(
                    tf.keras.Sequential([
                        iden() if pool_type is None else upconv(n),
                        deconv(kernel, n, stride=stride),
                        norm(norm_type),
                        act(act_type),
                        deconv(kernel, n, stride=1),
                        norm(norm_type),
                        act(act_type),
                    ]))
                self.is_contracting.append(False)
                spatsize_change = stride
                if pool_type is not None:
                    spatsize_change *= 2
                self.spatsize_changes.append(spatsize_change)
            prev_n = n
        # Spatial res. should come back to the original by now; a final
        # 1x1 conv to aggregate info and ensure desired # output channels
        self.layers.append(conv(1, n_feat[-1], stride=1))
        self.is_contracting.append(False)
        self.spatsize_changes.append(1)
        spatsizes = np.cumprod(self.spatsize_changes)
        assert spatsizes[-1] == 1, \
            "Resolution doesn't return to the original value"
