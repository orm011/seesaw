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

class ROIIndex(AccessMethod):
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
        coarse_df = coarse_df.reset_index(drop=True)
        #coarse_df = coarse_df.rename(columns={"clip_feature":"vectors",}) 
        #assert coarse_df.dbidx.is_monotonic_increasing, "sanity check"
        embedded_dataset = coarse_df["clip_feature"].values.to_numpy()
        return ROIIndex(
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
        '''
        fullmeta = self.vector_meta[self.vector_meta.dbidx.isin(included)]
        nframes = len(included)
        dbidxs = np.zeros(nframes) * -1
        dbscores = np.zeros(nframes)
        activations = []
        for i, (dbidx, frame_vec_meta) in enumerate(fullmeta.groupby("dbidx")):
            dbidxs[i] = dbidx
            boxscs = np.zeros(frame_vec_meta.shape[0])
            for j in range(frame_vec_meta.shape[0]): 
                tup = frame_vec_meta.iloc[j : j + 1]
                # GET BOX
                # GET IMAGE

                # GET VECTOR
                image_vector = tup.vectors.values[0]
                # CROSS VECTOR
                #print(tup)
                #print(tup.vectors.values[0])
                score = image_vector @ vector.reshape(-1)
                boxscs[j] = score
            frame_activations = frame_vec_meta.assign(score=boxscs)
            frame_activations = frame_activations[frame_activations.score == frame_activations.score.max()][
                ["x1", "y1", "x2", "y2", "dbidx", "score", "filename"]#["x1", "y1", "x2", "y2", "_x1", "_y1", "_x2", "_y2", "dbidx", "score", "filename"]
            ]
            activations.append(frame_activations)
            dbscores[i] = np.max(boxscs)

        topkidx = np.argsort(-dbscores)[:topk]
        

        return {
            "dbidxs": dbidxs[topkidx].astype("int"),
            "nextstartk": 100, #nextstartk,
            "activations": [activations[idx] for idx in topkidx],
        }
        '''

    def new_query(self):
        return ROIQuery(self)

    def subset(self, indices: pr.BitMap):
        mask = self.vector_meta.dbidx.isin(indices)
        if mask.all(): 
            return self

        return ROIIndex(
            embedding=self.embedding,
            vectors=self.vectors[mask],
            vector_meta=self.vector_meta[mask].reset_index(drop=True),
        )

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

def get_pos_negs_all_v2(dbidxs, label_db: LabelDB, vec_meta: pd.DataFrame):
    idxs = pr.BitMap(dbidxs)
    relvecs = vec_meta[vec_meta.dbidx.isin(idxs)]

    pos = []
    neg = []
    for idx in dbidxs:
        acc_vecs = relvecs[relvecs.dbidx == idx]
        acc_boxes = acc_vecs[["x1", "x2", "y1", "y2"]]
        label_boxes = label_db.get(idx, format="df")
        ious = box_iou(label_boxes, acc_boxes)
        total_iou = ious.sum(axis=0)
        negatives = total_iou == 0
        negvec_positions = acc_vecs.index[negatives].values

        # get the highest iou positives for each
        max_ious_id = np.argmax(ious, axis=1)
        max_ious = np.max(ious, axis=1)

        pos_idxs = pr.BitMap(max_ious_id[max_ious > 0])
        # if label_boxes.shape[0] > 0: # some boxes are size 0 bc. of some bug in the data, so don't assert here.
        #     assert len(pos_idxs) > 0

        posvec_positions = acc_vecs.index[pos_idxs].values
        pos.append(posvec_positions)
        neg.append(negvec_positions)

    posidxs = pr.BitMap(np.concatenate(pos))
    negidxs = pr.BitMap(np.concatenate(neg))
    return posidxs, negidxs

class ROIQuery(InteractiveQuery):
    def __init__(self, db: ROIIndex):
        super().__init__(db)

    def getXy(self):
        pos, neg = get_pos_negs_all_v2(
            self.label_db.get_seen(), self.label_db, self.index.vector_meta
        )
        
        allpos = self.index.vectors[pos]
        allneg = self.index.vectors[neg]
        Xt = np.concatenate([allpos, allneg])
        yt = np.concatenate([np.ones(len(allpos)), np.zeros(len(allneg))])
        return Xt, yt
