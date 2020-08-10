from os.path import dirname, join
import numpy as np

from ..log import get_logger
logger = get_logger()

from .. import const, os as xm_os
from ..imprt import preset_import
from .general import _savefig


def matrix_as_image(arr, outpath=None, gamma=None):
    """Visualizes an array into an image.

    By putting minimum (across all channels) at 0 and maximum at
    ``dtype_max``.

    Args:
        arr (numpy.ndarray): Array to be transformed into an image. Can be
            H-by-W (grayscale) or H-by-W-by-3/4 (RGB or RGBA).
        outpath (str, optional): Where to visualize the result to. ``None``
            means ``os.path.join(const.Dir.tmp, 'matrix_as_image.png')``.
        gamma (float, optional): For gamma correction.

    Raises:
        ValueError: If ``arr`` is neither 2D or 3D.

    Writes
        - An image of the matrix.
    """
    from ..img import gamma_correct
    cv2 = preset_import('cv2')

    if outpath is None:
        outpath = join(const.Dir.tmp, 'matrix_as_image.png')

    dtype = 'uint8'
    dtype_max = np.iinfo(dtype).max

    arr = arr.astype(float)
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape + (1,))
    elif arr.ndim == 3:
        assert (arr.shape[-1] in (1, 3, 4)), \
            ("Only single- (grayscale), three- (RGB), "
             "or four-channel (RGBA) matrices are supported")
    else:
        raise ValueError("'arr' needs to be either 2D or 3D")

    n_chs = arr.shape[-1]
    if n_chs == 4:
        arr, a = arr[:, :, :3], arr[:, :, 3]
        assert a.min() >= 0 and a.max() == 1, "Alpha must be [0, 1]"
        im_a = (a * dtype_max).astype(dtype)
    else:
        a = None

    minv, maxv = arr.min(), arr.max()
    if minv == maxv:
        im = (arr * dtype_max).astype(dtype)
        logger.warning(
            ("RGB channels contain only a single value: %f, so only "
             "operations performed: multiplied by dtype_max and cast to "
             "integer"), maxv)
    else:
        im = (dtype_max * (arr - minv) / (maxv - minv)).astype(dtype)
        # astype() safe only because we know it's [0, dtype_max]

    if a is not None:
        im = np.dstack((im, im_a))

    if gamma is not None:
        im = gamma_correct(im, gamma)

    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    if im.shape[-1] == 4:
        # RGBA
        cv2.imwrite(outpath, im[:, :, [2, 1, 0, 3]]) # OpenCV uses BGR
    else:
        # Grayscale or RGB
        cv2.imwrite(outpath, im[:, :, ::-1])

    logger.info("Matrix visualized as image to:\n\t%s", outpath)


def matrix_as_heatmap_complex(*args, **kwargs):
    """Wraps :func:`matrix_as_heatmap` for complex number support.

    Just pass in the parameters that :func:`matrix_as_heatmap` takes.
    ``'_mag'`` and ``'_phase'`` will be appended to ``outpath`` to produce the
    magnitude and phase heatmaps, respectively. Specifically, magnitude is
    computed by :func:`numpy.absolute`, and phase by :func:`numpy.angle`.

    Writes
        - A magnitude heatmap with ``'_mag'`` in its filename.
        - A phase heatmap with ``'_phase'`` in its filename.
    """
    outpath = kwargs.get('outpath', None)
    if outpath is None:
        outpath = join(const.Dir.tmp, 'matrix_as_heatmap_complex.png')
    for suffix in ('mag', 'phase'):
        l = outpath.split('.')
        l[-2] += '_' + suffix
        kwargs['outpath'] = '.'.join(l)
        args_l = []
        for i, x in enumerate(args):
            if i == 0: # mat
                if suffix == 'mag':
                    args_l.append(np.absolute(x))
                else:
                    args_l.append(np.angle(x))
            else:
                args_l.append(x)
        args = tuple(args_l)
        matrix_as_heatmap(*args, **kwargs)


def matrix_as_heatmap(mat, cmap='viridis', center_around_zero=False,
                      outpath=None, contents_only=False, figtitle=None):
    """Visualizes a matrix as heatmap.

    Args:
        mat (numpy.ndarray): Matrix to visualize as heatmp. May contain
            NaN's, which will be plotted white.
        cmap (str, optional): Colormap to use.
        center_around_zero (bool, optional): Whether to center colorbar around
            0 (so that zero is no color, i.e., white). Useful when matrix
            consists of both positive and negative values, and 0 means
            "nothing". ``None`` means default colormap and auto range.
        outpath (str, optional): Path to which the visualization is saved to.
            ``None`` means ``os.path.join(const.Dir.tmp,
            'matrix_as_heatmap.png')``.
        contents_only (bool, optional): Whether to plot only the contents
            (i.e., no borders, axes, etc.). If ``True``, the heatmap will be
            of exactly the same size as your matrix, useful when you want to
            plot heatmaps separately and later concatenate them into a single
            one.
        figtitle (str, optional): Figure title. ``None`` means no title.

    Raises:
        ValueError: If ``mat`` has wrong dimensions.

    Writes
        - A heatmap of the matrix.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ok_version = '2.0.2'
    if matplotlib.__version__ != ok_version:
        logger.warning(
            ("Developed and tested with Matplotlib %s (you are using %s). "
             "Known to be buggy with 3.0.0"),
            ok_version, matplotlib.__version__)

    if outpath is None:
        outpath = join(const.Dir.tmp, 'matrix_as_heatmap.png')

    if mat.ndim == 2:
        pass
    elif mat.ndim == 3:
        if mat.shape[2] == 1:
            mat = np.squeeze(mat)
        else:
            raise ValueError(
                "If `mat` is 3D, the 3rd dimension must have a single channel")
    else:
        raise ValueError("`mat` must be 2D or \"nominally\" 3D")
    mat = mat.astype(float)

    # Figure
    dpi = 96 # assumed
    w_in = mat.shape[1] / dpi
    h_in = mat.shape[0] / dpi
    if contents_only:
        # Output heatmap will have the exact same shape as input matrix
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w_in, h_in)
    else:
        plt.figure(figsize=(w_in, h_in))

    # Axis
    if contents_only:
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        ax = plt.gca()

    # Set title
    if (not contents_only) and (figtitle is not None):
        ax.set_title(figtitle)

    if center_around_zero:
        v_abs_max = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
        v_max, v_min = v_abs_max, -v_abs_max
        im = ax.imshow(mat, cmap=cmap, interpolation='none',
                       vmin=v_min, vmax=v_max)
    else:
        im = ax.imshow(mat, interpolation='none')

    if not contents_only:
        # Colorbar
        # Create an axes on the right side of ax; width will be 4% of ax,
        # and the padding between cax and ax will be fixed at 0.1 inch
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        plt.colorbar(im, cax=cax)

    # Make directory, if necessary
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)

    # Save plot
    _savefig(outpath, contents_only=contents_only, dpi=dpi)

    plt.close('all')

    logger.info("Matrix visualized as heatmap to:\n\t%s", outpath)
