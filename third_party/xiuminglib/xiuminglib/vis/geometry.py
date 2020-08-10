from os.path import join
import numpy as np

from .. import const
from ..imprt import preset_import


def ptcld_as_isosurf(pts, out_obj, res=128, center=False):
    """Visualizes point cloud as isosurface of its TDF.

    Args:
        pts (array_like): Cartesian coordinates in object space, of shape
            N-by-3.
        out_obj (str): The output path of the surface .obj.
        res (int, optional): Resolution of the TDF.
        center (bool, optional): Whether to center these points around object
            space origin.

    Writes
        - The .obj file of the isosurface.
    """
    from skimage.measure import marching_cubes_lewiner
    from trimesh import Trimesh
    from trimesh.io.export import export_mesh
    from ..geometry.ptcld import ptcld2tdf

    # Point cloud to TDF
    tdf = ptcld2tdf(pts, res=res, center=center)

    # Isosurface of TDF
    vs, fs, ns, _ = marching_cubes_lewiner(
        tdf, 0.999 / res, spacing=(1 / res, 1 / res, 1 / res))

    mesh = Trimesh(vertices=vs, faces=fs, normals=ns)
    export_mesh(mesh, out_obj)


def normal_as_image(normal_map, alpha_map, outpath=None):
    """Visualizes the normal map by converting vectors to pixel values.

    The background is black, complying with industry standards (e.g.,
    Adobe AE).

    Args:
        normal_map (numpy.ndarray): H-by-W-by-3 array of normal vectors.
        alpha_map (numpy.ndarray): H-by-W array of alpha values.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means ``os.path.join(const.Dir.tmp,
            'normal_as_image.png')``.

    Writes
        - The normal image.
    """
    cv2 = preset_import('cv2')

    if outpath is None:
        outpath = join(const.Dir.tmp, 'normal_as_image.png')

    dtype = 'uint8'
    dtype_max = np.iinfo(dtype).max

    # [-1, 1]
    im = (normal_map / 2 + 0.5) * dtype_max
    # [0, dtype_max]

    bg = np.zeros(im.shape)
    alpha = np.dstack([alpha_map] * 3)
    im = np.multiply(alpha, im) + np.multiply(1 - alpha, bg)

    cv2.imwrite(outpath, im.astype(dtype)[..., ::-1])


def depth_as_image(depth_map, alpha_map, outpath=None):
    """Visualizes a(n) (aliased) depth map and an (anti-aliased) alpha map
    as a single depth image.

    Output has black background, with bright values for closeness to the
    camera. If the alpha map is anti-aliased, the result depth map will
    be nicely anti-aliased.

    Args:
        depth_map (numpy.ndarray): 2D array of (aliased) raw depth values.
        alpha_map (numpy.ndarray): 2D array of (anti-aliased) alpha
            values.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means ``os.path.join(const.Dir.tmp,
            'depth_as_image.png')``.

    Writes
        - The (anti-aliased) depth image.
    """
    cv2 = preset_import('cv2')

    if outpath is None:
        outpath = join(const.Dir.tmp, 'depth_as_image.png')

    dtype = 'uint8'
    dtype_max = np.iinfo(dtype).max

    # Cap background depth at the object's maximum depth
    is_fg = depth_map < depth_map.max()
    max_val = depth_map[is_fg].max()
    depth_map[depth_map > max_val] = max_val

    min_val = depth_map.min()
    im = dtype_max * (max_val - depth_map) / (max_val - min_val)
    # Now [0, dtype_max]

    # Anti-aliasing
    bg = np.zeros(im.shape)
    im = np.multiply(alpha_map, im) + np.multiply(1 - alpha_map, bg)

    cv2.imwrite(outpath, im.astype(dtype))
