# pylint: disable=relative-beyond-top-level

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from .seq import Network as BaseNetwork
from ..util import logging as logutil


logger = logutil.Logger(loggee="networks/mlp")


class Network(BaseNetwork):
    def __init__(self, widths, act=None, skip_at=None):
        super().__init__()
        depth = len(widths)
        if act is None:
            act = [None] * depth
        assert len(act) == depth, \
            "If not `None`, `act` must have the save length as `widths`"
        for w, a in zip(widths, act):
            if isinstance(a, str):
                a = tf.keras.layers.Activation(a)
            layer = tf.keras.layers.Dense(w, activation=a)
            self.layers.append(layer)
        self.skip_at = skip_at

    def __call__(self, x):
        # Shortcircuit if simply sequential
        if self.skip_at is None:
            return super().__call__(x)
        # Need to concatenate input at some levels
        x_ = x + 0 # make a copy
        for i, layer in enumerate(self.layers):
            y = layer(x_)
            if i in self.skip_at:
                y = tf.concat((y, x), -1)
            x_ = y
        return y
