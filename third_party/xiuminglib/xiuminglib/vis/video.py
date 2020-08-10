from os.path import join, dirname
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..log import get_logger
logger = get_logger()

from .. import const
from ..io import img as imgio
from ..os import makedirs
from ..imprt import preset_import


def make_apng(
        imgs, labels=None, label_top_left_xy=(100, 100), font_size=100,
        font_color=(1, 0, 0), font_ttf=None, duration=1, outpath=None):
    r"""Writes a list of (optionally labeled) images into an animated PNG.

    Args:
        imgs (list(numpy.ndarray or str)): An image is either a path or an
            array (mixing ok, but arrays will need to be written to a temporary
            directory). If array, should be of type ``uint`` and of shape H-by-W
            (grayscale) or H-by-W-by-3 (RGB).
        labels (list(str), optional): Labels used to annotate the images.
        label_top_left_xy (tuple(int), optional): The XY coordinate of the
            label's top left corner.
        font_size (int, optional): Font size.
        font_color (tuple(float), optional): Font RGB, normalized to
            :math:`[0,1]`.
        font_ttf (str, optional): Path to the .ttf font file. Defaults to Arial.
        duration (float, optional): Duration of each frame in seconds.
        outpath (str, optional): Where to write the output to (a .apng file).
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_apng.apng')``.

    Raises:
        TypeError: If any input image is neither a string nor an array.

    Writes
        - An animated PNG of the images.
    """
    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_apng.apng')
    if not outpath.endswith('.apng'):
        outpath += '.apng'
    makedirs(dirname(outpath))

    # Font
    if font_ttf is None:
        font = ImageFont.truetype(const.Path.open_sans_regular, font_size)
    else:
        gfile = preset_import('gfile')
        open_func = open if gfile is None else gfile.Open
        with open_func(font_ttf, 'rb') as h:
            font_bytes = BytesIO(h.read())
        font = ImageFont.truetype(font_bytes, font_size)

    def put_text(img, text):
        dtype_max = np.iinfo(np.array(img).dtype).max
        color = tuple(int(x * dtype_max) for x in font_color)
        ImageDraw.Draw(img).text(label_top_left_xy, text, fill=color, font=font)
        return img

    imgs_loaded = []
    for img_i, img in enumerate(imgs):
        if isinstance(img, str):
            # Path
            img = imgio.load(img)
            if labels is not None:
                img = put_text(img, labels[img_i])
            imgs_loaded.append(img)
        elif isinstance(img, np.ndarray):
            # Array
            assert np.issubdtype(img.dtype, np.unsignedinteger), \
                "If image is provided as an array, it has to be `uint`"
            if (img.ndim == 3 and img.shape[2] == 1) or img.ndim == 2:
                img = np.dstack([img] * 3)
            img = Image.fromarray(img)
            if labels is not None:
                img = put_text(img, labels[img_i])
            imgs_loaded.append(img)
        else:
            raise TypeError(type(img))

    duration = duration * 1000 # because in ms

    gfile = preset_import('gfile')
    open_func = open if gfile is None else gfile.Open
    with open_func(outpath, 'wb') as h:
        imgs_loaded[0].save(
            h, save_all=True, append_images=imgs_loaded[1:],
            duration=duration)

    logger.info("Images written as an animated PNG to:\n\t%s", outpath)


def make_video(
        imgs, fps=24, outpath=None, matplotlib=True, dpi=96, bitrate=-1):
    """Writes a list of images into a grayscale or color video.

    Args:
        imgs (list(numpy.ndarray)): Each image should be of type ``uint8`` or
            ``uint16`` and of shape H-by-W (grayscale) or H-by-W-by-3 (RGB).
        fps (int, optional): Frame rate.
        outpath (str, optional): Where to write the video to (a .mp4 file).
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_video.mp4')``.
        matplotlib (bool, optional): Whether to use ``matplotlib``.
            If ``False``, use ``cv2``.
        dpi (int, optional): Dots per inch when using ``matplotlib``.
        bitrate (int, optional): Bit rate in kilobits per second when using
            ``matplotlib``; reasonable values include 7200.

    Writes
        - A video of the images.
    """
    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_video.mp4')
    makedirs(dirname(outpath))

    assert imgs, "Frame list is empty"
    h, w = imgs[0].shape[:2]
    for frame in imgs[1:]:
        assert frame.shape[:2] == (h, w), \
            "All frames must have the same shape"

    if matplotlib:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import animation

        w_in, h_in = w / dpi, h / dpi
        fig = plt.figure(figsize=(w_in, h_in))
        Writer = animation.writers['ffmpeg'] # may require you to specify path
        writer = Writer(fps=fps, bitrate=bitrate)

        def img_plt(arr):
            img_plt_ = plt.imshow(arr)
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            return img_plt_

        anim = animation.ArtistAnimation(fig, [(img_plt(x),) for x in imgs])
        anim.save(outpath, writer=writer)
        # If obscure error like "ValueError: Invalid file object: <_io.Buff..."
        # occurs, consider upgrading matplotlib so that it prints out the real,
        # underlying ffmpeg error

        plt.close('all')

    else:
        cv2 = preset_import('cv2')

        # TODO: debug codecs (see http://www.fourcc.org/codecs.php)
        if outpath.endswith('.mp4'):
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # fourcc = cv2.VideoWriter_fourcc(*'X264')
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            # fourcc = 0x00000021
        elif outpath.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise NotImplementedError("Video type of\n\t%s" % outpath)

        vw = cv2.VideoWriter(outpath, fourcc, fps, (w, h))

        for frame in imgs:
            if frame.ndim == 3:
                frame = frame[:, :, ::-1] # cv2 uses BGR
            vw.write(frame)

        vw.release()

    logger.info("Images written as a video to:\n%s", outpath)
