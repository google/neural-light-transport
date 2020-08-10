from os.path import exists, dirname
import numpy as np

from .. import log, os as xm_os
logger = log.get_logger()


def load_or_save_np(data_f, fallback=None):
    """Loads the data file if it exists. Otherwise, if fallback is provided,
    call fallback and save its return to disk.

    Args:
        data_f (str): Path to the data file, whose extension will be used for deciding
            how to load the data.
        fallback (function, optional): Fallback function used if data file doesn't exist.
            Its return will be saved to ``data_f`` for future loadings. It should not
            take arguments, but if yours requires taking arguments, just wrap yours with::

                fallback=lambda: your_fancy_func(var0, var1)

    Raises:
        NotImplementedError: If file extension is neither .npy nor .npz.

    Returns:
        Data loaded if ``data_f`` exists; otherwise, ``fallback``'s return
        (``None`` if no fallback).

    Writes
        - Return by the fallback, if provided.
    """
    # Decide data file type
    ext = data_f.split('.')[-1].lower()
    if ext == 'npy':
        load_func = np.load
        save_func = np.save
    elif ext == 'npz':
        load_func = np.load
        save_func = np.savez
    else:
        raise NotImplementedError(ext)

    # Load or call fallback
    if exists(data_f):
        data = load_func(data_f)
        msg = "Loaded: "
    else:
        msg = "File doesn't exist "
        if fallback is None:
            data = None
            msg += "(fallback not provided): "
        else:
            data = fallback()
            out_dir = dirname(data_f)
            xm_os.makedirs(out_dir)
            save_func(data_f, data)
            msg += "(fallback provided); fallback return now saved to: "
    msg += data_f

    logger.info(msg)
    return data
