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

"""An NLT-specific inference pipeline that first computes an average
observation feature map using multiple observations, and then runs
inference using this fixed, aggregated feature map.
"""

from os.path import join, basename
from glob import glob
from tqdm import tqdm
from absl import app, flags

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import datasets
import models
from util import io as ioutil, logging as logutil


flags.DEFINE_string(
    'ckpt', '/path/to/ckpt-100', "path to checkpoint (prefix only)")
flags.DEFINE_integer(
    'batch_size_override', None,
    "use a batch size different than what was used during training")
flags.DEFINE_integer(
    'n_obs_batches', 1,
    "number of observation batches used for the observation path")
flags.DEFINE_integer('fps', 24, "frames per second for the result video")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="nlt_test")


def get_config_ini():
    return '/'.join(FLAGS.ckpt.split('/')[:-2]) + '.ini'


def make_datapipe(mode, config):
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, mode)

    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch)
    return datapipe


def restore_model(config):
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config)

    model.register_trainable()

    # Resume from checkpoint
    assert model.trainable_registered, (
        "Register the trainable layers to have them restored from the "
        "checkpoint")
    ckpt = tf.train.Checkpoint(net=model)
    ckpt.restore(FLAGS.ckpt).expect_partial()

    return model


def infer(model, datapipe, feat_agg, outroot, report_every=10):
    batch_i = 0
    for batch in datapipe:
        outdir = join(outroot, 'batch{i:09d}'.format(i=batch_i))

        bs = batch[0].shape[0]
        obs_override = [tf.tile(x, (bs, 1, 1, 1)) for x in feat_agg]

        _, _, _, to_vis = model.call(batch, 'test', obs_override=obs_override)

        # Visualize
        outdir = outdir.format(i=batch_i)
        model.vis_batch(to_vis, outdir, 'test')

        batch_i += 1
        if batch_i % report_every == 0:
            logger.info("Done inferring %d batches", batch_i)


def extract_feat(model, datapipe):
    obs_feat_extractor = model.net['obs']

    if FLAGS.n_obs_batches > 0:
        datapipe = datapipe.take(FLAGS.n_obs_batches)
    datapipe = list(datapipe)

    # Extract features
    feat_agg = []
    for batch in tqdm(datapipe, desc="Extract features"):
        _, base, _, _, _, rgb, _, _, _, _, _ = batch
        # Forward through the observation path
        x = rgb - base
        feat = []
        for layer in obs_feat_extractor.layers:
            y = layer(x)
            feat.append(y)
            x = y
        # Concatenate
        if feat_agg:
            for level, level_feat in enumerate(feat):
                feat_agg[level] = tf.concat(
                    (feat_agg[level], level_feat), axis=0)
        else:
            feat_agg = feat # init. with first batch

    # Compute average feature using all extracted features
    feat_agg = [tf.reduce_mean(x, axis=0, keepdims=True) for x in feat_agg]

    # Each element is 1xHxWxC
    return feat_agg


def main(_):
    config_ini = get_config_ini()
    config = ioutil.read_config(config_ini)
    if FLAGS.batch_size_override is not None:
        config.set('DEFAULT', 'bs', str(FLAGS.batch_size_override))

    # Model
    model = restore_model(config)

    # Datasets
    datapipe_train = make_datapipe('train', config)
    datapipe_test = make_datapipe('test', config)

    # Extract features from observations
    feat_agg = extract_feat(model, datapipe_train)

    # Run inference on the test data
    outroot = join(
        config_ini[:-4], 'vis_test', basename(FLAGS.ckpt) + '_pred')
    infer(model, datapipe_test, feat_agg, outroot)

    # Compile all visualized batches into a consolidated view (e.g., an
    # HTML or a video)
    batch_vis_dirs = sorted(glob(join(outroot, '*')))
    outpref = outroot.rstrip('/') # proper extension should be added in the
    # user-overridden function below
    view_at = model.compile_batch_vis(
        batch_vis_dirs, outpref, 'test', fps=FLAGS.fps)
    logger.info("Compilation available for viewing at\n\t%s" % view_at)


if __name__ == '__main__':
    app.run(main)
