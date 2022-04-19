import argparse
import ray

from seesaw.dataset_manager import GlobalDataManager

if __name__ == "__main__":
    print("running script...")
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
        "--dataset_src",
        type=str,
        default=None,
        help="If cloning a dataset before preprocessing, name for source dataset",
    )
    parser.add_argument(
        "--seesaw_root", type=str, help="Seesaw root folder where dataset will live"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="String identifier for newly created dataset (will also be used as folder name)",
    )
    parser.add_argument("--model_path", type=str, help="path for model")
    # parser.add_argument('--image_archive', type=str, default=None, help='alternative path to images using an archive file')
    # parser.add_argument('--image_archive_prefix', type=str, default='', help='common prefix to use within archive')

    args = parser.parse_args()

    gdm = GlobalDataManager(args.seesaw_root)
    print("existing datasets: ", gdm.list_datasets())

    if args.image_src is not None:
        ds = gdm.create_dataset(
            image_src=args.image_src, dataset_name=args.dataset_name
        )
    elif args.dataset_src is not None:
        ds = gdm.clone(ds_name=args.dataset_src, clone_name=args.dataset_name)
    else:
        ds = gdm.get_dataset(args.dataset_name)

    print("connecting to ray...")
    ray.init("auto")
    print(ray.available_resources())
    ds.preprocess2(model_path=args.model_path)
