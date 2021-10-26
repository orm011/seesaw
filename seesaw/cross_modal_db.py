import torchvision
import torch
import pandas as pd
import os
import numpy as np
import PIL.Image
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import torchvision.transforms as T
import pyroaring as pr

from collections import deque, defaultdict
import sklearn
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import typing
import torch.utils.data
from .embeddings import XEmbedding

from .ui.widgets import MImageGallery

def show_images(db, idxs):
    db.get_
    MImageGallery()

class EmbeddingDB(object):
    """Structure holding a dataset,
     together with precomputed embeddings
     and (optionally) data structure
    """
    def __init__(self, raw_dataset : torch.utils.data.Dataset,
                 embedding : XEmbedding,
                 embedded_dataset : np.ndarray):
        self.raw = raw_dataset
        self.embedding = embedding
        all_indices = pr.BitMap()
        all_indices.add_range(0, len(self.raw))
        self.all_indices = pr.FrozenBitMap(all_indices)
#        self.urls = urls
        self.embedded = embedded_dataset 
     
        #norms = np.linalg.norm(embedded_dataset, axis=1)[:,np.newaxis]
        # if not np.isclose(norms,1.).all():
        #     print('Warning: db activations are not normalized {}'.format(norms.max()))

        # self.embedded_raw = embedded_dataset  # not necesarily normalized
        # self.embedded = embedded_dataset / (norms + 1e-5)
        assert len(self.raw) == self.embedded.shape[0]
        # index = NearestNeighbors(metric='cosine')
        # index.fit(self.embedded)
        # self.index = index
    
    def __len__(self):
        return len(self.raw)

    def query(self, *, topk, mode, cluster_id=None, vector=None, exclude=None, return_scores=False):
        if exclude is None:
            exclude = pr.BitMap([])        
        included = pr.BitMap(self.all_indices).difference(exclude)
        if len(included) == 0:
            return np.array([]),np.array([])

        if len(included) <= topk:
            topk = len(included)

        if vector is None:
            assert mode == 'random'
        elif vector is not None:
            assert mode in ['nearest', 'dot']
        else:
            assert False
            
        vecs = self.embedded[included]        

        if mode == 'nearest':
            scores = sklearn.metrics.pairwise.cosine_similarity(vector, vecs)
            scores = scores.reshape(-1)
        elif mode == 'dot':
            scores = vecs @ vector.reshape(-1)
        elif mode == 'random':
            scores = np.random.randn(vecs.shape[0])
        elif mode == 'model':
            with torch.no_grad():
                scores = model.forward(torch.from_numpy(self.embedded[included].astype('float32')))
                scores = scores.numpy()[:,1]      

        maxpos = np.argsort(-scores)[:topk]
        dbidxs = np.array(included)[maxpos]
        scores = scores[maxpos]

        ret = dbidxs
        assert ret.shape[0] == scores.shape[0]
        sret = pr.BitMap(ret)
        assert len(sret) == ret.shape[0]  # no repeats
        assert ret.shape[0] == topk  # return quantity asked, in theory could be less
        assert sret.intersection_cardinality(exclude) == 0  # honor exclude request

        return ret, scores

import  sklearn.metrics.pairwise

class HEmbeddingDB(EmbeddingDB):
    """Structure holding a dataset,
     together with precomputed embeddings
     and (optionally) data structure
    """
    def __init__(self, db, clusters=None):
        super().__init__(db.raw, db.embedding, db.embedded)
        if clusters is None:
            clusters = np.ones(len(db))
        self.clusters = clusters
        self.indexes = {}
        self.dbidx = {}
        
        cluster_vals = np.unique(clusters)
        for cl in cluster_vals:
            dbidxs = np.where(clusters == cl)[0]
            index = NearestNeighbors(metric='cosine')
            index.fit(self.embedded[dbidxs])
            self.indexes[cl] = index
            self.dbidx[cl] = dbidxs

    def query(self, *, topk, mode, cluster_id=None, vector=None, model = None, exclude=None, return_scores=False):
        if cluster_id is None:
            # index = self.index
            dbidx_table = np.arange(self.embedded.shape[0]).astype('int')
        else:
            # index = self.indexes[cluster_id]
            dbidx_table = self.dbidx[cluster_id]

        # breakpoint()
        if exclude is None:
            exclude = pr.BitMap([])
        
        included = pr.BitMap(dbidx_table).difference(exclude)
        if len(included) == 0:
            if return_scores:
                return np.array([]),np.array([])
            else:
                return np.array([])

        if len(included) <= topk:
            topk = len(included)

        if vector is None and model is None:
            assert mode == 'random'
        elif vector is not None:
            assert mode in ['nearest', 'dot']
        elif model is not None:
            assert mode in ['model']
        else:
            assert False
            
        ## todo: change nearest to be less inefficient
        vecs = self.embedded[included]
        if mode == 'nearest':
            scores = sklearn.metrics.pairwise.cosine_similarity(vector, vecs)
            scores = scores.reshape(-1)
        elif mode == 'dot':
            scores = vecs @ vector.reshape(-1)
        elif mode == 'random':
            scores = np.random.randn(vecs.shape[0])
        elif mode == 'model':
            with torch.no_grad():
                scores = model.forward(torch.from_numpy(self.embedded[included].astype('float32')))
                scores = scores.numpy()[:,1]      

        maxpos = np.argsort(-scores)[:topk]
        dbidxs = np.array(included)[maxpos]
        scores = scores[maxpos]

        ret = dbidxs
        assert ret.shape[0] == scores.shape[0]
        sret = pr.BitMap(ret)
        assert len(sret) == ret.shape[0]  # no repeats
        assert ret.shape[0] == topk  # return quantity asked, in theory could be less
        assert sret.intersection_cardinality(exclude) == 0  # honor exclude request

        if return_scores:
            return ret, scores
        else:
            return ret