import pandas as pd
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import pyroaring as pr

import torch.utils.data
from .embeddings import XEmbedding
from .query_interface import *
from .dataset_manager import GlobalDataManager

class CoarseIndex(AccessMethod):
    """Structure holding a dataset,
     together with precomputed embeddings
     and (optionally) data structure
    """
    def __init__(self, images : torch.utils.data.Dataset,
                 embedding : XEmbedding,
                 vectors : np.ndarray, 
                 vector_meta : pd.DataFrame):
        self.images = images
        self.embedding = embedding
        self.vectors = vectors
        self.vector_meta = vector_meta

        self.all_indices = pr.FrozenBitMap(self.vector_meta.dbidx.values)
        #assert len(self.images) >= self.vectors.shape[0]
    
    def string2vec(self, string : str) -> np.ndarray:
        init_vec = self.embedding.from_string(string=string)
        init_vec = init_vec/np.linalg.norm(init_vec)
        return init_vec

    @staticmethod
    def from_path(gdm : GlobalDataManager, dataset_name : str, index_subpath : str, model_name :str):
        dm = gdm.get_dataset(dataset_name)
        images = dm.get_pytorch_dataset()
        embedding = gdm.get_model_actor(model_name)
        
        coarse_meta_path= f'{gdm.root}/{index_subpath}/vectors.coarse.cached'

        assert os.path.exists(coarse_meta_path)
        coarse_df = pd.read_parquet(coarse_meta_path)
        assert coarse_df.dbidx.is_monotonic_increasing, 'sanity check'
        embedded_dataset = coarse_df['vectors'].values.to_numpy()
        vector_meta = coarse_df.drop('vectors', axis=1)
        return CoarseIndex(images=images, embedding=embedding, vectors=embedded_dataset, vector_meta=vector_meta)
    
    def query(self, *, topk, mode, vector=None, exclude=None, startk=None):
        if exclude is None:
            exclude = pr.BitMap([])        
        included = pr.BitMap(self.all_indices).difference(exclude)
        if len(included) == 0:
            return np.array([]),np.array([])

        if len(included) <= topk:
            topk = len(included)

        assert mode == 'dot'
        vecs = self.vectors[included]        
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

    def new_query(self):
        return CoarseQuery(self)

class CoarseQuery(InteractiveQuery):
    def __init__(self, db : CoarseIndex):
        super().__init__(db)

    def getXy(self, idxbatch, box_dict):
        idxbatch = np.array([self.db.all_indices.rank(idx) - 1 for idx in idxbatch])
        Xt = self.db.vectors[idxbatch]
        yt = np.array([box_dict[idx].shape[0] > 0 for idx in idxbatch])
        return Xt,yt