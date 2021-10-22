#import pickle
# import cloudpickle
# pickle.Pickler = cloudpickle.Pickler
#import multiprocessing as mp
import argparse

import seesaw
from seesaw import GlobalDataManager, trace_emb_jit
import ray    

if __name__ == '__main__':
    #mp.set_start_method('spawn', True)
    parser = argparse.ArgumentParser(description='create and preprocess dataset for use by Seesaw')
    parser.add_argument('--image_src', type=str, default='', help='If creating a new dataset, folder where to find images')
    parser.add_argument('--dataset_src', type=str, default='', help='If cloning a dataset before preprocessing, name for source dataset')
    parser.add_argument('--seesaw_root', type=str, help='Seesaw root folder where dataset will live')
    parser.add_argument('--dataset_name', type=str, help='String identifier for newly created dataset (will also be used as folder name)')
    parser.add_argument('--model_path', type=str, help='path to use for jitted model')
#    parser.add_argument('--glob_patterns', type=str, default='', help='glob patterns to use to search for images')
    args = parser.parse_args()

    gdm = GlobalDataManager(args.seesaw_root)

    if args.image_src != '':
        ds = gdm.create_dataset(image_src=args.image_src, dataset_name=args.dataset_name)
    else:
        ds = gdm.clone(ds_name=args.dataset_src, clone_name=args.dataset_name)

    ray.init('auto')
    print(ray.available_resources())
    ds.preprocess2(model_path=args.model_path)