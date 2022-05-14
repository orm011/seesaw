import argparse

parser = argparse.ArgumentParser(description="preprocess dataset for use by Seesaw")
parser.add_argument(
    "--multiscale_path",
    type=str,
    required=True,
    help="path for multiscale index this will be  based on",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="where to store the needed output",
)

args = parser.parse_args()

import ray

ray.init("auto", namespace="seesaw")

from .preprocessor import from_fine_grained

from_fine_grained(args.multiscale_path, args.output_path)
