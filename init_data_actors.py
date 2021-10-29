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

## make the embedding model itself a separate actor
# pass handle all the way down.
# I want calls to be synchronous“
class ModelService(XEmbedding):
    def __init__(self, model_ref):
        self.model_ref = model_ref

    def from_string(self, *args, **kwargs):
        return ray.get(self.model_ref.from_string.remote(*args, **kwargs))

    def from_image(self, *args, **kwargs):
        return ray.get(self.model_ref.from_image.remote(*args, **kwargs))

    def from_raw(self, *args,  **kwargs):
        return ray.get(self.model_ref.from_raw.remote(*args, **kwargs))

class DB(object):
    evref : ray.ObjectRef
    def __init__(self, dataset_loader, dbsample=None):
        print('loading ev...') 
        ev = dataset_loader()
        print('loaded ev...now putting it into shared store')
        self.ev_ref = ray.put(ev)
        print('put ev into store')
        del ev # don't keep local ref

        assert dbsample is None

        ## do this at client side
        # if dbsample is not None:
        #     ev = extract_subset(ev0, idxsample=np.sort(dbsample))
        #self.hdb = EmbeddingDB(ev.image_dataset,ev.embedding,ev.embedded_dataset)
        print('inited db...') 
    
    def ready(self):
        return True

    def get_ev(self):
        return self.ev_ref

    def get_image_paths(self, idxs):
        paths = self.ev.image_dataset.paths[idxs]
        from_root = self.ev.image_dataset.root + '/' + paths
        return from_root

    def embed_raw(self, data):
        return self.hdb.embedding.from_raw(data)
    
    # def get_boxes(self, category):
    #     c = category
    #     assert category in self.ev.query_ground_truth.columns
    #     box_cats = self.ev.box_data.category.unique()         
    #     if c not in box_cats: # NB box bounds only work for bdd... fix before using example
    #         dbidxs = np.where(self.ev.query_ground_truth[c] > 0)[0]
    #         boxdf = pd.DataFrame({'dbidx':dbidxs})
    #         boxes = boxdf.assign(x1=0, y1=0, x2=1280, y2=720, category=c)
    #     else:
    #         boxes = self.ev.box_data[self.ev.box_data.category == c]
    #     return boxes
    
    def get_vectors(self, idxs=None):
        if idxs is None:
            return self.hdb.embedded
        else:
            return self.hdb.embedded[idxs]

    def query(self, *, topk, mode, cluster_id=None, vector=None, 
              model = None, exclude=None, return_scores=False):
        return self.hdb.query(topk=topk, mode=mode, cluster_id=cluster_id, 
                              vector=vector, exclude=exclude, return_scores=return_scores)

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

        dbactor = RemoteDB.options(name=k).remote(dataset_loader=loader, dbsample=None)
        dbs.append(dbactor)
        handles.append(dbactor.ready.remote())

    handles.append(model_actor.ready.remote())
    ray.get(handles)
    print('db loaded and model ready')

    while input() != 'exit':
        pass
    print('exiting...')