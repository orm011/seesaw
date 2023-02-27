from pydantic import BaseModel
import os
import numpy as np
from seesaw.knn_graph import get_weight_matrix, rbf_kernel
from .loop_base import *
from .util import clean_path
from ..research.knn_methods import KNNGraph, LabelPropagationRanker2, SimpleKNNRanker

from ..indices.multiscale.multiscale_index import _get_top_dbidxs, rescore_candidates

import pyroaring as pr
import scipy.sparse as sp
from ..services import _cache_closure

from ray.data.extensions import TensorArray


class WeightMatrixOptions(BaseModel):
    knn_path : str
    knn_k : int
    edist : float
    self_edges : bool
    normalized_weights : bool
    symmetric : bool
    xlx_matrix : bool = False


def lookup_weight_matrix(opts : WeightMatrixOptions, *,  use_cache : bool, X_vectors=None) -> sp.csr_array:
    key = opts.json()
    if opts.xlx_matrix:
        assert opts.symmetric
        assert not opts.self_edges

    def init():
        print(f'init weight matrix {opts=}')
        knng = KNNGraph.from_file(opts.knn_path)
        knng = knng.restrict_k(k=opts.knn_k)
        wm = get_weight_matrix(knng.knn_df, 
                            kfun=rbf_kernel(opts.edist),
                            self_edges=opts.self_edges, 
                            normalized=opts.normalized_weights,
                            symmetric=opts.symmetric, 
                            laplacian=opts.xlx_matrix)

        if opts.xlx_matrix:
            assert X_vectors is not None
            total_sum = wm.diagonal().sum()
            L = wm / total_sum        
            wm = X_vectors.T @ (L @ X_vectors)

        return wm

    return _cache_closure(init, key=key, use_cache=use_cache)

def get_weight_matrix_from_index(idx, weight_matrix_options, xlx_matrix=False):
    opts = WeightMatrixOptions(**weight_matrix_options)
    opts.xlx_matrix = xlx_matrix

    knn_path = clean_path(idx.get_knng_path(name=opts.knn_path))
    opts.knn_path = knn_path # replace with full path

    use_cache = True
    if knn_path.find('subset') > -1:
        use_cache = False
    weight_matrix = lookup_weight_matrix(opts, use_cache=use_cache, X_vectors=idx.vectors)
    return weight_matrix

def get_label_prop(q, label_prop_params):
    weight_matrix = get_weight_matrix_from_index(q.index, label_prop_params['matrix_options'])
    label_prop = LabelPropagationRanker2(weight_matrix=weight_matrix, **label_prop_params)
    return label_prop

class KnnProp2(LoopBase):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams, knn_model):
        super().__init__(gdm, q, params)
        self.state.knn_model = knn_model

    @staticmethod
    def from_params(gdm, q, p: SessionParams):
        knn_model = get_label_prop(q, p.interactive_options)
        return KnnProp2(gdm, q, p, knn_model)


    def set_text_vec(self, tvec):
        scores = self.q.index.score(tvec)
        self.state.knn_model.set_base_scores(scores)

    def next_batch(self):

        """
        gets next batch of image indices based on current vector
        """
        s = self.state
        p = self.params
        q = self.q

        sorted_idxs, sorted_scores = s.knn_model.top_k(k=None, unlabeled_only=True)
        candidates = _get_top_dbidxs(vec_idxs=sorted_idxs, scores=sorted_scores, 
                        vector_meta=q.index.vector_meta, exclude=q.returned, topk=p.shortlist_size)

        candidates = candidates.reset_index(drop=True)
        vector_meta = q.index.vector_meta

        fullmeta = vector_meta[vector_meta.dbidx.isin(pr.BitMap(candidates.dbidx.values))]
        vecs = TensorArray(q.index.vectors[fullmeta.index.values])
        fullmeta = fullmeta.assign(vectors=vecs, score=s.knn_model.current_scores()[fullmeta.index.values])
        ans =  rescore_candidates(fullmeta, topk=p.batch_size, **p.dict())
        self.q.returned.update(ans['dbidxs'])
        return ans

    def refine(self, change=None):
        # labels already added.
        # go over labels here since it takes time
        ## translating box labels to labels over the vector index.
        #### for each frame in a box label. box join with the vector index for that box.
        # seen_ids = np.array(self.q.label_db.get_seen())
        pos, neg = self.q.getXy(get_positions=True)
        idxs = np.concatenate([pos,neg])
        labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        s = self.state
        s.knn_model.update(idxs, labels)
