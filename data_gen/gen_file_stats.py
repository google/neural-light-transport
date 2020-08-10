"""This script globs all the input data and persists their paths as well as
existence to the disk, so that the training pipeline doesn't need to glob
the data during training.

This helps to safeguard against the training pipeline reading from non-existent
paths (caused by, e.g., failed rendering jobs).

Example Usage:

    python "$ROOT"/neural-light-transport/data_gen/gen_file_stats.py \
        --data_root="$ROOT"'/output/render/dragon_specular_imh512_uvs512_spp256/' \
        --out_json="$ROOT"/output/render/dragon_specular_imh512_uvs512_spp256.json
"""

from argparse import ArgumentParser
from os.path import join, exists, basename, relpath
from glob import glob
from tqdm import tqdm

from util import dump_json


parser = ArgumentParser(description="")
parser.add_argument(
    '--data_root', type=str, required=True, help="data root directory")
parser.add_argument(
    '--out_json', type=str, required=True, help="path to the result JSON")


def main(args):
    stats = {}

    for config_dir in tqdm(
            sorted(glob(join(args.data_root, '*'))),
            desc="Camera-light configurations"):
        id_ = basename(config_dir)

        stats[id_] = {
            'alpha': join(config_dir, 'alpha.png'),
            'cam': join(config_dir, 'cam.json'),
            'cvis': join(config_dir, 'cvis.png'),
            'diffuse': join(config_dir, 'diffuse.png'),
            'light': join(config_dir, 'light.json'),
            'lvis': join(config_dir, 'lvis.png'),
            'nn': join(config_dir, 'nn.json'),
            'rgb': join(config_dir, 'rgb.png'),
            'rgb_camspc': join(config_dir, 'rgb_camspc.png'),
            'uv2cam': join(config_dir, 'uv2cam.npy')}

        # Check existence
        all_exist = True
        for _, v in stats[id_].items():
            all_exist = all_exist and exists(v)
        stats[id_]['complete'] = all_exist

        # Make the paths relative, to reduce the file size and make it
        # root-independent
        for k, v in stats[id_].items():
            if k != 'complete':
                stats[id_][k] = relpath(v, args.data_root)

    dump_json(stats, args.out_json)


if __name__ == '__main__':
    main(parser.parse_args())
