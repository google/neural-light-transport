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
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError


def get_convex_hull(pts):
    try:
        hull = ConvexHull(pts)
    except QhullError:
        hull = None
    return hull


def in_hull(hull, pts):
    verts = hull.points[hull.vertices, :]
    hull = Delaunay(verts)
    return hull.find_simplex(pts) >= 0


def rad2deg(rad):
    return 180 / np.pi * rad
