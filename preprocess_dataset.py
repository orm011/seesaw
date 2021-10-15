from seesaw import preprocess_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create and preprocess dataset for use by Seesaw')
    parser.add_argument('--image_src', type=str, help='Folder where to find images')
    parser.add_argument('--seesaw_root', type=str, help='Seesaw root folder where to store the metadata')
    parser.add_argument('--dataset_name', type=str, help='String identifier for newly created dataset (will also be used as folder name)')
    args = parser.parse_args()
    preprocess_dataset(seesaw_root=args.seesaw_root,image_src=args.image_src,dataset_name=args.dataset_name)
