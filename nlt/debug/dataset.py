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

import sys
from os.path import join, dirname
from absl import app

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

sys.path.append('../')
import datasets
from util import io as ioutil


def main(_):
    config_ini = join(dirname(__file__), '..', 'config', 'dragon_specular.ini')
    config = ioutil.read_config(config_ini)

    # Make training dataset
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'train')

    path = dataset.files[1]
    ret = dataset._load_data(path)

    # Iterate
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch)
    for batch_i, batch in enumerate(datapipe):
        from IPython import embed; embed()


if __name__ == '__main__':
    app.run(main)
