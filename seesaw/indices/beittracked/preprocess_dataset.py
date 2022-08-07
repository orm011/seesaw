import argparse
from seesaw.definitions import resolve_path
from seesaw.vector_index import build_annoy_idx

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

    parser.add_argument("--cpu", action="store_true", help="use cpu rather than GPU")
    parser.add_argument("--model_path", type=str, required=True, help="path for model")
    parser.add_argument("--start", type=int, help="which index to start at")
    parser.add_argument("--end", type=int, help="which index to end at")

    args = parser.parse_args()

    import ray
    from seesaw.dataset import SeesawDatasetManager
    from preprocessor import preprocess_beit_dataset

    ds = SeesawDatasetManager(args.dataset_path)
    preprocess_beit_dataset(
        ds, clip_model_path=args.model_path, cpu=args.cpu, output_path=args.output_path, start_index=args.start, end_index=args.end, 
    )

    
