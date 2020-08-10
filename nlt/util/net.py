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

import numpy as np


def gen_feat_n(min_n, max_n, final_n=3):
    """Generates a sequence of numbers that can be used as numbers of
    channels across the network, excluding the first layer that produces
    an original-resolution feature map.

    E.g., `[8, 16, 32, 64, 64, 32, 16, 8, 4, 3]`.
    """
    assert max_n >= min_n and max_n >= final_n, \
        ("Max number of channels must be greater than or equal to the final "
         "number of channel")

    n_ch = [2 ** i for i in range(
        int(np.log2(min_n)) + 1, int(np.log2(max_n)) + 1)]

    # Prepend min_n if need be
    if not n_ch or n_ch[0] != min_n:
        n_ch = [min_n] + n_ch

    # Append max_n if need be
    if not n_ch or n_ch[-1] != max_n:
        n_ch.append(max_n)

    # Append a reversed version of the current list
    n_ch += n_ch[::-1]

    # Decay to final_n
    n_ch += [2 ** i for i in range(
        int(np.log2(n_ch[-1])) - 1, int(np.log2(final_n)), -1)]

    # Pop away >final_n from the reversed version
    while True:
        if not n_ch or n_ch[-1] >= final_n:
            break
        n_ch.pop()

    # Append final_n
    n_ch.append(final_n)

    return n_ch
