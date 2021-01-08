# NLT: Data Generation

This folder provides the code for rendering your own data. You do not need this
if you use our rendered data (available in "Downloads -> Rendered Data" of
[the project page](http://nlt.csail.mit.edu)).

`render.py` is the core script that renders a given camera-light configuration.
`gen_render_params_expects.py` generates parameters that define different
camera-light configurations (arguments to `render.py`), for you to distribute
`render.py` over multiple machines or a render farm to render all
configurations in parallel. `get_neighbors.py` is the script that generated the
JSON files indicating the nearest neighbor for each camera/light in the metadata
.zip.

Scenes are specified in Blender and rendered with Cycles, Blender's built-in
physically-based rendering engine.


## Setup

You should use the Python bundled inside Blender, rather than that of your
system or environment.

Because of the breaking changes in Blender 2.8x, please run our code with
Blender 2.7x. More specifically, we used Blender 2.78c, available
[here](https://download.blender.org/release/Blender2.78/blender-2.78c-linux-glibc219-x86_64.tar.bz2).

1. Clone this repository:
    ```
    cd "$ROOT"
    git clone https://github.com/google/neural-light-transport.git
    ```

1. "Install" Blender-Python (the binaries are pre-built, so just download
   and unzip):
    ```
    cd "$WHERE_YOU_WANT_BLENDER_INSTALLED"

    # Download
    wget https://download.blender.org/release/Blender2.78/blender-2.78c-linux-glibc219-x86_64.tar.bz2

    # Unzip the pre-built binaries
    tar -xvjf blender-2.78c-linux-glibc219-x86_64.tar.bz2
    ```

1. Install the dependencies to this *Blender-bundled* Python:
    ```
    cd blender-2.78c-linux-glibc219-x86_64/2.78/python/bin

    # Install pip for THIS Blender-bundled Python
    curl https://bootstrap.pypa.io/get-pip.py | ./python3.5m

    # Use THIS pip to install other dependencies
    ./pip install Pillow
    ./pip install tqdm
    ./pip install ipython
    ./pip install numpy
    ./pip install opencv-python
    ```

1. Make sure this Python can locate `xiuminglib`:
    ```
    export PYTHONPATH="$ROOT"/neural-light-transport/third_party/xiuminglib/:$PYTHONPATH
    ```


## Rendering

There are header instructions in all the main scripts, but the general workflow
is as follows.

1. UV unwrap the object of interest in the Blender scene. This needs doing only
   once per scene, and the rendering workers will read this UV unwrapping from
   the disk.
    ```
    "$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
        --background \
        --python "$ROOT"/neural-light-transport/data_gen/uv_unwrap.py \
        -- \
        --scene="$ROOT"/data/scenes-v2/dragon_specular.blend \
        --object=object \
        --outpath="$ROOT"/data/scenes-v2/dragon_specular_uv.pickle
    ```

1. Make sure that you can render a single camera-light configuration:
    ```
    "$WHERE_YOU_INSTALLED_BLENDER"/blender-2.78c-linux-glibc219-x86_64/blender \
        --background \
        --python "$ROOT"/neural-light-transport/data_gen/render.py \
        -- \
        --scene="$ROOT"/data/scenes-v2/dragon_specular.blend \
        --cached_uv_unwrap="$ROOT"/data/scenes-v2/dragon_specular_uv.pickle \
        --cam_json="$ROOT"/data/trainvali_cams/P28R.json \
        --light_json="$ROOT"/data/trainvali_lights/L330.json \
        --cam_nn_json="$ROOT"/data/neighbors/cams.json \
        --light_nn_json="$ROOT"/data/neighbors/lights.json \
        --imh='512' \
        --uvs='1024' \
        --spp='256' \
        --outdir="$ROOT"/data/scenes-v2/dragon_specular_imh512_uvs1024_spp256/trainvali_000020852_P28R_L330
    ```

1. Generate all camera-light configurations (rendering jobs) you want to render:
    ```
    python "$ROOT"/neural-light-transport/data_gen/gen_render_params_expects.py \
        --mode='trainvali+test' \
        --scene="$ROOT"/data/scenes-v2/dragon_specular.blend \
        --cached_uv_unwrap="$ROOT"/data/scenes-v2/dragon_specular_uv.pickle \
        --trainvali_cams="$ROOT"'/data/trainvali_cams/*.json' \
        --test_cams="$ROOT"'/data/test_cams/*.json' \
        --trainvali_lights="$ROOT"'/data/trainvali_lights/*.json' \
        --test_lights="$ROOT"'/data/test_lights/*.json' \
        --cam_nn_json="$ROOT"/data/neighbors/cams.json \
        --light_nn_json="$ROOT"/data/neighbors/lights.json \
        --imh='512' \
        --uvs='1024' \
        --spp='256' \
        --outroot="$ROOT"/data/scenes-v2/dragon_specular_imh512_uvs1024_spp256/ \
        --jobdir="$ROOT"/tmp/specular
    ```
   Note that any Python can be used for this step, not necessarily the
   Blender-bundled Python, because there is no Blender-specific operation.

1. Distribute the rendering jobs to your render farm to render all camera-light
   configurations, depending on your infrastructure.

1. Postprocess the rendered data: (1) appoximating the albedo by averaging
   all camera-light configurations in the UV space, weighing it with different
   light visibilities to produce "diffuse bases," and finally resampling the
   resultant bases into different camera views; (2) globbing the data and
   dumping the file list to disk, such that the training pipeline can just load
   this tiny file to know immediately which camera-light configuration has
   missing data (caused by, e.g., failed rendering jobs) and therefore should
   be skipped (helpful especially when existence checks are slow for your
   filesystem).
    ```
    python "$ROOT"/neural-light-transport/data_gen/postproc.py \
        --data_root="$ROOT"/data/scenes-v2/dragon_specular_imh512_uvs1024_spp256/ \
        --out_json="$ROOT"/data/scenes-v2/dragon_specular_imh512_uvs1024_spp256.json
    ```
