import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess dataset for use by Seesaw")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="which dataset (list of files) to run on",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="where to store the needed output",
    )

    parser.add_argument(
        "--build_index",
        type=str,
        required=True,
        help="where to store the needed output",
    )

    parser.add_argument("--cpu", action="store_true", help="use cpu rather than GPU")
    parser.add_argument("--model_path", type=str, required=True, help="path for model")

    args = parser.parse_args()

    import ray
    from seesaw.dataset import SeesawDatasetManager
    from seesaw.multiscale.preprocessor import preprocess_dataset

    ray.init("auto", namespace="seesaw")

    ds = SeesawDatasetManager(args.dataset_path)
    preprocess_dataset(
        ds, model_path=args.model_path, cpu=args.cpu, output_path=args.output_path
    )

    # from seesaw.vector_index import build_annoy_idx
    # if args.build_index:
    #     build_annoy_idx()
