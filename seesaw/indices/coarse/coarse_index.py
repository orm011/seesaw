import pandas as pd
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import pyroaring as pr

import torch.utils.data
from ...models.embeddings import XEmbedding
from ...query_interface import *
from ...definitions import resolve_path


class CoarseIndex(AccessMethod):
    """Structure holding a dataset,
    together with precomputed embeddings
    and (optionally) data structure
    """

    def __init__(
        self, embedding: XEmbedding, vectors: np.ndarray, vector_meta: pd.DataFrame, path : str = None
    ):
        self.path = path
        self.embedding = embedding
        self.vectors = vectors
        self.vector_meta = vector_meta
        self.all_indices = pr.FrozenBitMap(self.vector_meta.dbidx.values)
        # assert len(self.images) >= self.vectors.shape[0]

    def string2vec(self, string: str) -> np.ndarray:
        init_vec = self.embedding.from_string(string=string)
        init_vec = init_vec / np.linalg.norm(init_vec)
        return init_vec

    @staticmethod
    def from_path(index_path: str):
        from ...services import get_parquet, get_model_actor

        index_path = resolve_path(index_path)
        model_path = os.readlink(f"{index_path}/model")
        embedding = get_model_actor(model_path)
        vector_path = f"{index_path}/vectors"
        coarse_df = get_parquet(vector_path)
        assert coarse_df.dbidx.is_monotonic_increasing, "sanity check"
        embedded_dataset = coarse_df["vectors"].values.to_numpy()
        vector_meta = coarse_df.drop("vectors", axis=1)
        return CoarseIndex(
            embedding=embedding, vectors=embedded_dataset, vector_meta=vector_meta, 
            path = index_path
        )

    def query(self, *, topk, vector=None, exclude=None, startk=None, **kwargs):
        if exclude is None:
            exclude = pr.BitMap([])
        included = pr.BitMap(self.all_indices).difference(exclude)
        if len(included) == 0:
            return np.array([]), np.array([])

        if len(included) <= topk:
            topk = len(included)

        metas = self.vector_meta.dbidx.isin(included)
        vecs = self.vectors[metas]

        if vector is None:
            scores = np.random.randn(vecs.shape[0])
        else:
            scores = vecs @ vector.reshape(-1)

        maxpos = np.argsort(-scores)[:topk]
        dbidxs = np.array(included)[maxpos]
        # metas = metas.iloc[maxpos][['x1', 'y1', ]]
        scores = scores[maxpos]

        ret = dbidxs
        assert ret.shape[0] == scores.shape[0]
        sret = pr.BitMap(ret)
        assert len(sret) == ret.shape[0]  # no repeats
        assert ret.shape[0] == topk  # return quantity asked, in theory could be less
        assert sret.intersection_cardinality(exclude) == 0  # honor exclude request

        def make_acc(sc, dbidx):
            return pd.DataFrame.from_records(
                [dict(x1=0, y1=0, x2=224, y2=224, dbidx=dbidx, score=sc)]
            )

        return {
            "dbidxs": ret,
            "nextstartk": len(exclude) + ret.shape[0],
            "activations": [make_acc(sc, dbidx) for (sc, dbidx) in zip(scores, ret)],
        }

    def new_query(self):
        return CoarseQuery(self)

    def subset(self, indices: pr.BitMap):
        mask = self.vector_meta.dbidx.isin(indices)
        return CoarseIndex(
            embedding=self.embedding,
            vectors=self.vectors[mask],
            vector_meta=self.vector_meta[mask].reset_index(drop=True),

        )


class CoarseQuery(InteractiveQuery):
    def __init__(self, db: CoarseIndex):
        super().__init__(db)

    def getXy(self, get_positions=False):
        positions = np.array(
            [self.index.all_indices.rank(idx) - 1 for idx in self.label_db.get_seen()]
        )

        if get_positions:
            assert False, 'implement me'

        Xt = self.index.vectors[positions]
        yt = np.array(
            [
                len(self.label_db.get(idx, format="box")) > 0
                for idx in self.label_db.get_seen()
            ]
        )
        return Xt, yt
