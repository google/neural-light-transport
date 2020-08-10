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
