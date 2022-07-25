import pandas as pd
import os
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import pyroaring as pr

import torch.utils.data
from ...models.embeddings import XEmbedding
from ...query_interface import *
from ...definitions import resolve_path

def box_iou(tup, boxes):
    b1 = torch.from_numpy(
        np.stack([tup.x1.values, tup.y1.values, tup.x2.values, tup.y2.values], axis=1)
    )
    bxdata = np.stack(
        [boxes.x1.values, boxes.y1.values, boxes.x2.values, boxes.y2.values], axis=1
    )
    b2 = torch.from_numpy(bxdata)
    ious = torchvision.ops.box_iou(b1, b2)
    return ious.numpy()

def augment_score2(tup, vec_meta, vecs, *, agg_method, rescore_method):
    assert callable(rescore_method)
    vec_meta = vec_meta.reset_index(drop=True)
    ious = box_iou(tup, vec_meta)
    vec_meta = vec_meta.assign(iou=ious.reshape(-1))
    max_boxes = vec_meta.groupby("zoom_level").iou.idxmax()
    max_boxes = max_boxes.sort_index(
        ascending=True
    )  # largest zoom level (zoomed out) goes last
    relevant_meta = vec_meta.iloc[max_boxes]
    relevant_iou = (
        relevant_meta.iou > 0
    )  # there should be at least some overlap for it to be relevant
    max_boxes = max_boxes[relevant_iou.values]
    rel_vecs = vecs[max_boxes]

    if agg_method == "avg_score":
        sc = rescore_method(rel_vecs)
        ws = np.ones_like(sc)
        fsc = ws.reshape(-1) @ sc.reshape(-1)
        fsc = fsc / ws.sum()
        return fsc
    elif agg_method == "avg_vector":
        merged_vec = rel_vecs.mean(axis=0, keepdims=True)
        merged_vec = merged_vec / np.linalg.norm(merged_vec)
        return rescore_method(merged_vec)
    else:
        assert False, f"unknown agg_method {agg_method}"

class BeitIndex(AccessMethod):
    """Structure holding a dataset,
    together with precomputed embeddings
    and (optionally) data structure
    """

    def __init__(
        self, embedding: XEmbedding, vectors: np.ndarray, vector_meta: pd.DataFrame
    ):
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
        from seesaw.services import get_parquet, get_model_actor

        index_path = resolve_path(index_path)
        model_path = os.readlink(f"{index_path}/model")
        #model_path = os.readlink("/home/gridsan/groups/fastai/omoll/seesaw_root2/models/clip-vit-base-patch32/")
        embedding = get_model_actor(model_path)
        vector_path = f"{index_path}/vectors"
        coarse_df = get_parquet(vector_path, columns=['dbidx', 'x1', 'y1', 'x2', 'y2', 'clip_feature'])
        #coarse_df = coarse_df.rename(columns={"clip_feature":"vectors",}) 
        #assert coarse_df.dbidx.is_monotonic_increasing, "sanity check"
        embedded_dataset = coarse_df["clip_feature"].values.to_numpy()
        return BeitIndex(
            embedding=embedding, vectors=embedded_dataset, vector_meta=coarse_df
        )

    def query(self, *, topk, vector, exclude=None, startk=None, **kwargs):
        agg_method = 'avg_score'
        if exclude is None:
            exclude = pr.BitMap([])
        included = pr.BitMap(self.all_indices).difference(exclude)
        if len(included) == 0:
            return np.array([]), np.array([])

        if len(included) <= topk:
            topk = len(included)

        scores = self.vectors @ vector.reshape(-1)
        vec_idxs = np.argsort(-scores)
        scores = scores[vec_idxs]
        topscores = self.vector_meta[['dbidx', 'x1', 'y1', 'x2', 'y2']].iloc[vec_idxs]
        topscores = topscores.assign(score=scores)
        topscores = topscores[topscores.dbidx.isin(included)]
        scores_by_video = (
            topscores.groupby('dbidx').score.max().sort_values(ascending=False)
        )
        score_cutoff = scores_by_video.iloc[topk - 1]
        topscores = topscores[topscores.score >= score_cutoff]
        dbidxs = topscores.dbidx.unique()[:topk]
        activations = []
        for idx in dbidxs: 
            full_meta = topscores[topscores.dbidx == idx]
            activations.append(full_meta)
        return {
            "dbidxs": dbidxs,
            "nextstartk": 100, #nextstartk,
            "activations": activations,
        }

    def new_query(self):
        return BeitQuery(self)

    def subset(self, indices: pr.BitMap):
        mask = self.vector_meta.dbidx.isin(indices)
        if mask.all(): 
            return self

        return BeitIndex(
            embedding=self.embedding,
            vectors=self.vectors[mask],
            vector_meta=self.vector_meta[mask].reset_index(drop=True),
        )


class BeitQuery(InteractiveQuery):
    def __init__(self, db: BeitIndex):
        super().__init__(db)

    def getXy(self):
        positions = np.array(
            [self.index.all_indices.rank(idx) - 1 for idx in self.label_db.get_seen()]
        )
        Xt = self.index.vectors[positions]
        yt = np.array(
            [
                len(self.label_db.get(idx, format="box")) > 0
                for idx in self.label_db.get_seen()
            ]
        )
        return Xt, yt
