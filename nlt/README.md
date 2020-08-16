# NLT: Training and Testing

This folder is a general pipeline for TensorFlow 2 (eager) model training,
validation, and testing.

`trainvali.py` and `test.py` are the main scripts with the
training/validation/testing loops. `config/` contains the example configuration
files, from which the pipeline parses arguments. The NLT model is specified in
`models/nlt.py`, and the NLT data loader is specified in `datasets/nlt.py`.


## Setup

TensorFlow versions are specified in `../environment.yml`. The cuDNN version we
used is 7.6, and the CUDA version 10.1 (for TensorFlow-cuDNN/CUDA version
compatibility, see
[this Stack Overflow answer](https://stackoverflow.com/a/50622526/2106753)).

1. Clone this repository:
    ```
    cd "$ROOT"
    git clone https://github.com/google/neural-light-transport.git
    ```

1. Install a Conda environment with all dependencies:
    ```
    cd "$ROOT"/neural-light-transport
    conda env create -f environment.yml
    conda activate nlt
    ```

1. Make sure Python can locate third-party libraries:
    ```
    export PYTHONPATH="$ROOT"/neural-light-transport/third_party/:$PYTHONPATH
    export PYTHONPATH="$ROOT"/neural-light-transport/third_party/xiuminglib/:$PYTHONPATH
    ```

### Tips

* The IPython dependency in `environment.yml` is for `IPython.embed()` alone.
  If you are not using that to insert breakpoints during debugging, you can
  take it off. That said, it should not hurt to just leave it there.
* We would caution you against installing TensorFlow with Conda in this project.
  As seen in `environment.yml`, we ended up using pip instead for all three
  TensorFlow-related installations after much struggle with the version
  compatibility issues while installing with Conda.


## Training

Run the training and validation pipeline:
```
CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT"/neural-light-transport/nlt/trainvali.py \
    --config='dragon_specular.ini'
```

To visualize the losses and get handy links to the visualization webpages:
```
tensorboard --logdir "$EXPERIMENT_DIR" --bind_all
```
The visualization webpages (whose links can be found under the "TEXT" tab
in TensorBoard) include animated comparisons of the NLT predictions against
the diffuse bases and ground truth.

### About Distributed Training

The distributed training feature should work, but was not heavily tested.
Therefore, you may be better off just using a single GPU by making only, say,
GPU 3 visible:
```
CUDA_VISIBLE_DEVICES='3' python ...
```

### Tips

* Even if eager execution is used throughout, the data pipeline still runs in
  the graph mode (by design). This may complicate debugging the data pipeline
  in that you may not be able to insert breakpoints. Take a look at
  `debug/dataset.py`, where we call the dataset functions outside of the
  pipeline, insert breakpoints there, and debug.
* For easier debugging, consider commenting off `@tf.function` for
  `distributed_train_step()` in `trainvali.py`, but be sure to use this
  decorator for speed during training.
* This training plus validation pipeline is in fact general, so you can use it
  for totally different models and/or datasets: simply add them to `models/`
  and/or `datasets/` and follow the same API.


## Testing

Run the testing pipeline:
```
CUDA_VISIBLE_DEVICES="$GPU" \
    python "$ROOT"/neural-light-transport/nlt/nlt_test.py \
    --ckpt="$ROOT"'/output/train/lr:1e-3_mgm:-1/checkpoints/ckpt-43'
```
which runs inference with the given checkpoint on all test data, and eventually
produces a video visualization whose frames correspond to different camera-light
configurations.
