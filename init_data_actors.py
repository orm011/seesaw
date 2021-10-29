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
    ray.init('auto', namespace='seesaw')

    model_actor = ray.get_actor('clip')
    xclip = ModelService(model_actor)

    gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/data')
    ds_names = ['objectnet']#, 'bdd', 'coco', 'dota', 'lvis']

    dbs = []
    handles = []
    for k in ds_names:
        def loader():
            return load_ev(gdm=gdm, dsname=k, xclip=xclip, load_ground_truth=False, load_coarse=True)

        dbactor = RemoteDB.options(name=k).remote(dataset_loader=loader)
        dbs.append(dbactor)
        handles.append(dbactor.ready.remote())

    handles.append(model_actor.ready.remote())
    ray.get(handles)
    print('db loaded and model ready')

    while input() != 'exit':
        pass
    print('exiting...')