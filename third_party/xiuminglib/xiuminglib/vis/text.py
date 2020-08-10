from os.path import dirname
import numpy as np

from ..log import get_logger
logger = get_logger()

from .. import os as xm_os
from ..imprt import preset_import


def text_as_image(
        text, imsize=256, thickness=2, dtype='uint8', outpath=None,
        quiet=False):
    """Rasterizes a text string into an image.

    The text will be drawn in white to the center of a black canvas.
    Text size gets automatically figured out based on the provided
    thickness and image size.

    Args:
        text (str): Text to be drawn.
        imsize (float or tuple(float), optional): Output image height and width.
        thickness (float, optional): Text thickness.
        dtype (str, optional): Image type.
        outpath (str, optional): Where to dump the result to. ``None``
            means returning instead of writing it.
        quiet (bool, optional): Whether to refrain from logging.
            Effective only when ``outpath`` is not ``None``.

    Returns or Writes
        - An image of the text.
    """
    cv2 = preset_import('cv2')

    if isinstance(imsize, int):
        imsize = (imsize, imsize)
    assert isinstance(imsize, tuple), \
        "`imsize` must be an int or a 2-tuple of ints"

    # Unimportant constants not exposed to the user
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    base_bgr = (0, 0, 0) # black
    text_bgr = (1, 1, 1) # white

    # Base canvas
    dtype_max = np.iinfo(dtype).max
    im = np.tile(base_bgr, imsize + (1,)).astype(dtype) * dtype_max

    # Figure out the correct font scale
    font_scale = 1 / 128 # real small
    while True:
        (text_width, text_height), bl_y = cv2.getTextSize(
            text, font_face, font_scale, thickness)
        if bl_y + text_height >= imsize[0] or text_width >= imsize[1]:
            # Undo the destroying step before breaking
            font_scale /= 2
            (text_width, text_height), bl_y = cv2.getTextSize(
                text, font_face, font_scale, thickness)
            break
        font_scale *= 2

    # Such that the text is at the center
    bottom_left_corner = (
        (imsize[1] - text_width) // 2,
        (imsize[0] - text_height) // 2 + text_height)
    cv2.putText(
        im, text, bottom_left_corner, font_face, font_scale,
        [x * dtype_max for x in text_bgr], thickness)

    if outpath is None:
        return im

    # Write
    outdir = dirname(outpath)
    xm_os.makedirs(outdir)
    cv2.imwrite(outpath, im)

    if not quiet:
        logger.info("Text rasterized into image to:\n%s", outpath)
