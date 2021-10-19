import pickle
# import cloudpickle
# pickle.Pickler = cloudpickle.Pickler

import multiprocessing as mp
import argparse

import seesaw
from seesaw import GlobalDataManager

    
if __name__ == '__main__':
    mp.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='create and preprocess dataset for use by Seesaw')
    parser.add_argument('--image_src', type=str, default='', help='If creating a new dataset, folder where to find images. Leave out to use existin dataset')
    parser.add_argument('--dataset_src', type=str, default='', help='If cloning a dataset before preprocessing, name for source datset')
    parser.add_argument('--seesaw_root', type=str, help='Seesaw root folder where to store the metadata')
    parser.add_argument('--dataset_name', type=str, help='String identifier for newly created dataset (will also be used as folder name)')
    args = parser.parse_args()

    gdm = GlobalDataManager(args.seesaw_root)

    if args.image_src != '':
        ds = gdm.create_dataset(image_src=args.image_src, dataset_name=args.dataset_name)
    else:
        ds = gdm.clone(ds_name=args.dataset_src, clone_name=args.dataset_name)

    ds.preprocess()