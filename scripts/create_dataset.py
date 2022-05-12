import argparse
from seesaw.dataset import create_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create and preprocess dataset for use by Seesaw"
    )
    parser.add_argument(
        "--image_src",
        type=str,
        default=None,
        help="If creating a new dataset, folder where to find images",
    )
    parser.add_argument(
        "--output_path", type=str, help="Folder where dataset will live"
    )

    args = parser.parse_args()
    ds = create_dataset(image_src=args.image_src, output_path=args.output_path)
