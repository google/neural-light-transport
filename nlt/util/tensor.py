# pylint: disable=relative-beyond-top-level

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from . import logging as logutil


logger = logutil.Logger(loggee="util/tensor")


def shape_as_list(x):
    return x.get_shape().as_list()


def make_nhwc(batch, c=3):
    """Makes a NxHxW(x1) tensor NxHxWxC, written in graph mode.
    """
    # Assert 3D or 4D
    n_dims = tf.rank(batch)
    assert_op = tf.debugging.Assert(
        tf.logical_or(tf.equal(n_dims, 3),
                      tf.equal(n_dims, 4)), [n_dims])

    # If necessary, 3D to 4D
    with tf.control_dependencies([assert_op]):
        batch = tf.cond(
            tf.equal(n_dims, 4),
            true_fn=lambda: batch,
            false_fn=lambda: tf.expand_dims(batch, -1))

    # Repeat the last channel Cx, after asserting #channels is 1
    shape = tf.shape(batch)
    assert_op = tf.debugging.Assert(tf.equal(shape[3], 1), [shape])
    with tf.control_dependencies([assert_op]):
        return tf.tile(batch, (1, 1, 1, c))
