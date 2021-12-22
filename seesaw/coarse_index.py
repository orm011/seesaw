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
from .query_interface import AccessMethod


class CoarseIndex(AccessMethod):
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
        self.embedded = embedded_dataset 
        assert len(self.raw) == self.embedded.shape[0]
    
    def __len__(self):
        return len(self.raw)

    def query(self, *, topk, mode, vector=None, exclude=None, startk=None):
        if exclude is None:
            exclude = pr.BitMap([])        
        included = pr.BitMap(self.all_indices).difference(exclude)
        if len(included) == 0:
            return np.array([]),np.array([])

        if len(included) <= topk:
            topk = len(included)

        assert mode == 'dot'
        vecs = self.embedded[included]        
        if vector is None:
            scores = np.random.randn(vecs.shape[0])
        else:
            scores = vecs @ vector.reshape(-1)

        maxpos = np.argsort(-scores)[:topk]
        dbidxs = np.array(included)[maxpos]
        scores = scores[maxpos]

        ret = dbidxs
        assert ret.shape[0] == scores.shape[0]
        sret = pr.BitMap(ret)
        assert len(sret) == ret.shape[0]  # no repeats
        assert ret.shape[0] == topk  # return quantity asked, in theory could be less
        assert sret.intersection_cardinality(exclude) == 0  # honor exclude request

        return ret, len(exclude) + ret.shape[0]