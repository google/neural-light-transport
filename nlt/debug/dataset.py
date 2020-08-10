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
