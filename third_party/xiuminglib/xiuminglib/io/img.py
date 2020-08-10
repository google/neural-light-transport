from os.path import dirname
import numpy as np
from PIL import Image

from ..log import get_logger
logger = get_logger()

from ..imprt import preset_import
from ..os import makedirs


def load(path, as_array=False):
    """Loads an image.

    Args:
        path (str): Path to the image file.
        as_array (bool, optional): Whether to return the image as an array.
            Defaults to ``False``.

    Returns:
        A PIL image type or numpy.ndarray: Loaded image.
    """
    gfile = preset_import('gfile')
    open_func = open if gfile is None else gfile.Open
    with open_func(path, 'rb') as h:
        img = Image.open(h)
        img.load()

    logger.debug("Image loaded from:\n\t%s", path)

    if as_array:
        return np.array(img)
    return img


def write_img(arr_uint, outpath):
    r"""Writes an ``uint`` array/image to disk.

    Args:
        arr_uint (numpy.ndarray): A ``uint`` array.
        outpath (str): Output path.

    Writes
        - The resultant image.
    """
    if arr_uint.ndim == 3 and arr_uint.shape[2] == 1:
        arr_uint = np.dstack([arr_uint] * 3)

    img = Image.fromarray(arr_uint)

    # Write to disk
    gfile = preset_import('gfile')
    open_func = open if gfile is None else gfile.Open
    makedirs(dirname(outpath))
    with open_func(outpath, 'wb') as h:
        img.save(h)

    logger.debug("Image written to:\n\t%s", outpath)


def write_arr(arr_0to1, outpath, img_dtype='uint8', clip=False):
    r"""Writes an array to disk as an image.

    Args:
        arr_0to1 (numpy.ndarray): Array with values roughly :math:`\in [0,1]`.
        outpath (str): Output path.
        img_dtype (str, optional): Image data type. Defaults to ``'uint8'``.
        clip (bool, optional): Whether to clip values to :math:`[0,1]`.
            Defaults to ``False``.

    Writes
        - The resultant image.

    Returns:
        numpy.ndarray: The resultant image array.
    """
    if clip:
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    assert arr_0to1.min() >= 0 and arr_0to1.max() <= 1, \
        "Input should be in [0, 1], or allow it to be clipped"

    # Float array to image
    img_arr = (arr_0to1 * np.iinfo(img_dtype).max).astype(img_dtype)

    write_img(img_arr, outpath)

    return img_arr
