# pylint: disable=relative-beyond-top-level

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from util import logging as logutil
from .base import Network as BaseNetwork


logger = logutil.Logger(loggee="networks/seq")


class Network(BaseNetwork):
    """Assuming simple sequential flow.
    """
    def build(self, input_shape):
        seq = tf.keras.Sequential(self.layers)
        seq.build(input_shape)
        for layer in self.layers:
            assert layer.built, "Some layers not built"

    def __call__(self, tensor):
        x = tensor
        for layer in self.layers:
            y = layer(x)
            x = y
        return y
