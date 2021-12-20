import ray
import torch.optim
import sys
from ray.util.multiprocessing import Pool
import inspect
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import copy
import logging
import pyroaring as pr
import time
import functools

import seesaw
from seesaw import *

class DB(object):
    evref : ray.ObjectRef
    def __init__(self, dataset_loader):
        print('loading ev...') 
        ev = dataset_loader()
        print('loaded ev...now putting it into shared store')
        self.ev_ref = ray.put(ev)
        print('put ev into store')
        del ev # don't keep local ref
        print('inited db...') 
    
    def ready(self):
        return True

    def get_ev(self):
        return self.ev_ref

RemoteDB = ray.remote(DB)

import argparse

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_ground_truth", action='store_true')
    parser.add_argument("--load_coarse_embedding", action='store_true')
    parser.add_argument("--datasets", action="extend", nargs="+", type=str, default=[])
    parser.add_argument("--namespace", type=str, default='seesaw')
    args = parser.parse_args()

    ray.init('auto', namespace=args.namespace)


    def get_model(dsname):
        d = {'panama_frames_finetune4':'birdclip', 'bird_guide_finetuned':'birdclip', 'bird_guide_224_finetuned':'birdclip'}
        s = d.get(dsname, 'clip')
        name =  f'{s}#actor'
        model_actor = ray.get_actor(name)
        xclip = ModelService(model_actor)
        return xclip

    gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/data')
    ds_names = args.datasets

    dbs = []
    handles = []
    for k in ds_names:
        def loader():
            return load_ev(gdm=gdm, dsname=k, xclip=get_model(k), 
                    load_ground_truth=args.load_ground_truth, 
                    load_coarse=args.load_coarse_embedding)

        dbactor = RemoteDB.options(name=f'{k}#actor', lifetime='detached').remote(dataset_loader=loader)
        dbs.append(dbactor)
        handles.append(dbactor.ready.remote())

    # handles.append(model_actor.ready.remote())
    ray.get(handles)
    print('db loaded and model ready...')