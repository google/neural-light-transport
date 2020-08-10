# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def sample_pdf(val, weights, n_samples, det=False, eps=1e-5):
    weights += eps # prevent NaN's
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat((tf.zeros_like(cdf[:, :1]), cdf), -1)
    if det:
        u = tf.linspace(0., 1., n_samples)
        u = tf.broadcast_to(u, cdf.shape[:-1] + (n_samples,))
    else:
        u = tf.random.uniform(cdf.shape[:-1] + (n_samples,))
    # Invert CDF
    ind = tf.searchsorted(cdf, u, side='right') # (n_rays, n_samples)
    below = tf.maximum(0, ind - 1)
    above = tf.minimum(ind, cdf.shape[-1] - 1)
    ind_g = tf.stack((below, above), -1) # (n_rays, n_samples, 2)
    cdf_g = tf.gather(cdf, ind_g, axis=-1, batch_dims=len(ind_g.shape) - 2)
    val_g = tf.gather(val, ind_g, axis=-1, batch_dims=len(ind_g.shape) - 2)
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0] # (n_rays, n_samples)
    denom = tf.where(denom < eps, tf.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom
    samples = val_g[:, :, 0] + t * (val_g[:, :, 1] - val_g[:, :, 0])
    return samples # (n_rays, n_samples)
