import numpy as np

from ..linalg import project_onto, normalize


def project_onto_plane(pts, v1, v2, v3):
    """Projects 3D points onto a 2D plane defined by three vertices.

    Since the 2D space can freely rotate and translate on the 3D plane, the
    first vertex will be used as the origin, and the vector from the first to
    the second vertex will be the x-axis.

    WIP.
    """
    pts = np.array(pts)
    v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)

    # Get +x, +y, and +z directions
    dx = v2 - v1
    dx = normalize(dx)
    dz = np.cross(dx, v3 - v1)
    dz = normalize(dz)
    dy = np.cross(dz, dx)
    dy = normalize(dy)

    # Express new x-, y-, and z-axes in terms of the old system
    x = v1 + dx
    y = v1 + dy
    z = v1 + dz

    # Project the points to this new system
    proj_x = project_onto(pts, x)
    proj_y = project_onto(pts, y)
    proj_z = project_onto(pts, z)
    # FIXME: unfinished


def ptcld2tdf(pts, res=128, center=False):
    """Converts point cloud to truncated distance function (TDF).

    Maximum distance is capped at 1 / ``res``.

    Args:
        pts (array_like): Cartesian coordinates in object space. Of shape
            N-by-3.
        res (int, optional): Resolution of the TDF.
        center (bool, optional): Whether to center these points around the
            object space origin.

    Returns:
        numpy.ndarray: Output TDF.
    """
    pts = np.array(pts)

    n_pts = pts.shape[0]

    if center:
        pts_center = np.mean(pts, axis=0)
        pts -= np.tile(pts_center, (n_pts, 1))

    tdf = np.ones((res, res, res)) / res
    cnt = np.zeros((res, res, res))

    # -0.5 to 0.5 in every dimension
    extent = 2 * np.abs(pts).max()
    pts_scaled = pts / extent

    # Compute distance from center of each involved voxel to its surface
    # points
    for i in range(n_pts):
        pt = pts_scaled[i, :]
        ind = np.floor((pt + 0.5) * (res - 1)).astype(int)
        v_ctr = (ind + 0.5) / (res - 1) - 0.5
        dist = np.linalg.norm(pt - v_ctr)
        n = cnt[ind[0], ind[1], ind[2]]
        tdf[ind[0], ind[1], ind[2]] = \
            (tdf[ind[0], ind[1], ind[2]] * n + dist) / (n + 1)
        cnt[ind[0], ind[1], ind[2]] += 1

    return tdf
