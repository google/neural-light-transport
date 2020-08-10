# pylint: disable=relative-beyond-top-level

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from util import logging as logutil


logger = logutil.Logger(loggee="networks/base")


class Network:
    def __init__(self):
        self.layers = []

    def __call__(self, x):
        raise NotImplementedError

    @staticmethod
    def str2none(str_):
        """Mostly to overcome there being no `config.getnone()` method.
        """
        assert isinstance(str_, str), "Call this only on strings"
        if str_.lower() == 'none':
            return None
        return str_
