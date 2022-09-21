from seesaw.util import parallel_read_parquet
import numpy as np
import pyroaring as pr
from ray.data.extensions import TensorArray
import pynndescent 
import pandas as pd

import os
import scipy.sparse


def get_knn_matrix(knn_df, reverse=False):
    ## num vertices
    n_vertex = max(knn_df.src_vertex.max(), knn_df.dst_vertex.max()) + 1
    
    entries = knn_df.distance.values
    row_index = knn_df.src_vertex.values
    column_index = knn_df.dst_vertex.values
    
    if reverse:
        tmp = column_index
        row_index = column_index
        column_index = tmp
        
    return scipy.sparse.csr_matrix((entries, (row_index, column_index)), shape=(n_vertex,n_vertex))


def get_rev_lookup_ranges(rev_df, nvecs):
    cts = rev_df.dst_vertex.value_counts().sort_index()
    counts_filled = cts.reindex(np.arange(-1, nvecs), fill_value=0)
    ind_ptr = counts_filled.cumsum().values
    return ind_ptr

class KNNGraph:
    def __init__(self, knn_df, rev_df=None):
        self.knn_df = knn_df
        actual_nvecs = knn_df.src_vertex.max() + 1
        self.nvecs = actual_nvecs

        if rev_df is None:
            rev_df = self.knn_df.sort_values('dst_vertex')

        self.k = knn_df.dst_rank.max() + 1        
        self.knn_df = knn_df
        self.rev_df = rev_df
        self.ind_ptr = get_rev_lookup_ranges(rev_df, self.nvecs)

    def restrict_k(self, *, k):
        if k < self.k:
            knn_df = self.knn_df.query(f'dst_rank < {k}').reset_index(drop=False)
            rev_df = self.rev_df.query(f'dst_rank < {k}').reset_index(drop=False)
            return KNNGraph(knn_df, rev_df=rev_df)
        elif k > self.k:
            assert False, f'can only do up to k={self.k} neighbors based on input df'
        else:
            return self

    def make_symmetric(self):
        knndf = self.knn_df
        knndf_reciprocal = knndf.assign(src_vertex=knndf.dst_vertex, dst_vertex=knndf.src_vertex)
        symknn_all = pd.concat([knndf, knndf_reciprocal], ignore_index=True)
        symknn_all = symknn_all.sort_values('src_vertex').drop_duplicates().reset_index(drop=True)
        return KNNGraph(knn_df=symknn_all)

    @staticmethod
    def from_vectors(vectors, *, n_neighbors, n_jobs=-1, low_memory=False, **kwargs):
        """ returns a graph and also the index """
        index2 = pynndescent.NNDescent(vectors, n_neighbors=n_neighbors+1, metric='dot', n_jobs=n_jobs, low_memory=low_memory, **kwargs)
        positions, distances = index2.neighbor_graph
        identity = (positions == np.arange(positions.shape[0]).reshape(-1,1))
        any_identity = identity.sum(axis=1) > 0
        exclude = identity
        exclude[~any_identity, -1] = 1 # if there is no identity in the top k+1, exclude the k+1
        assert (exclude.sum(axis=1) == 1).all()
        positions1 = positions[~exclude].reshape(-1,n_neighbors   )
        distances1 = distances[~exclude].reshape(-1,n_neighbors)

        nvec, nneigh = positions1.shape
        iis, jjs = np.meshgrid(np.arange(nvec), np.arange(nneigh), indexing='ij')

        knn_df = pd.DataFrame({ 'src_vertex':iis.reshape(-1),
                                'dst_vertex':positions1.reshape(-1), 
                                'distance':distances1.reshape(-1),
                                'dst_rank':jjs.reshape(-1)
                                
                                })
        
        knn_graph = KNNGraph(knn_df)
        return knn_graph, index2

                
    def save(self, path, num_blocks=10, overwrite=False):
        import ray.data
        import shutil

        if os.path.exists(path) and overwrite:
            shutil.rmtree(path)

        os.makedirs(path, exist_ok=False)

        ds = ray.data.from_pandas(self.knn_df).lazy()
        ds.repartition(num_blocks=num_blocks).write_parquet(f'{path}/forward.parquet')

        ds2 = ray.data.from_pandas(self.rev_df).lazy()
        ds2.repartition(num_blocks=num_blocks).write_parquet(f'{path}/backward.parquet')

    
    @staticmethod
    def from_file(path, parallelism=0):
        # if not cache:
        df = parallel_read_parquet(f'{path}/forward.parquet', parallelism=parallelism)
        # from ..services import get_parquet
        # df = get_parquet()
        rev_df = parallel_read_parquet(f'{path}/backward.parquet', parallelism=parallelism)
        return KNNGraph(df, rev_df=rev_df)

    def rev_lookup(self, dst_vertex) -> pd.DataFrame:
        return self.rev_df.iloc[self.ind_ptr[dst_vertex]:self.ind_ptr[dst_vertex+1]]

from scipy.special import expit as sigmoid

class SimpleKNNRanker:
    def __init__(self, knng, init_scores=None):
        self.knng : KNNGraph = knng

        if init_scores is None:
            self.init_numerators = np.ones(self.knng.nvecs)*.1 # base if nothing is given
        else:
            self.set_base_scores(init_scores)

        self.pscount = 1.
        
        self.numerators = np.zeros_like(self.init_numerators)
        self.denominators = np.zeros_like(self.init_numerators)

        self.labels = np.zeros_like(self.init_numerators)
        self.is_labeled = np.zeros_like(self.init_numerators)
        
        self.all_indices = pr.FrozenBitMap(range(self.knng.nvecs))
        
    def current_scores(self):
        num = self.pscount*self.init_numerators + self.numerators
        denom = self.pscount + self.denominators
        estimates = num/denom
        return self.labels*self.is_labeled + estimates*(1-self.is_labeled)
        
    def set_base_scores(self, scores):
        assert self.knng.nvecs == scores.shape[0]
        self.init_numerators = sigmoid(2*scores)

    def update(self, idx, label):
        idx = int(idx)
        label = float(label)
        
        assert np.isclose(label,0) or np.isclose(label,1)
        
        if self.is_labeled[idx] > 0: # if new label for old 
            old_label = self.labels[idx]
            delta_denom = 0
            delta_num = label - old_label # erase old label and add new label
        else:
            delta_num = label
            delta_denom = 1
        
        self.labels[idx] = label
        self.is_labeled[idx] = 1
                
        ## update scores for all v such that idx \in knn(v)
        rev_neighbors = self.knng.rev_lookup(idx).src_vertex.values
        # rev_weights = 
        self.numerators[rev_neighbors] += delta_num
        self.denominators[rev_neighbors] += delta_denom
        
    def top_k(self, k, unlabeled_only=True):
        if unlabeled_only:
            subset = np.where(self.is_labeled < 1)[0]
        else: 
            subset = np.array(self.all_indices)
            
        raw_scores = self.current_scores()
        
        topk_positions = np.argsort(-raw_scores[subset])[:k]
        topk_indices = subset[topk_positions]
        
        return topk_indices, raw_scores[topk_indices]


import scipy.sparse as sp
from scipy.special import expit as sigmoid
import numpy as np


def kernel(cosine_distance, edist):
    # edist = cosine distance needed to reduce neighbor weight to 1/e.
    # for cosine sim, the similarity ranges from -1 to 1. 
    spread = 1./edist
    return np.exp(-cosine_distance*spread)


def prepare(knng : KNNGraph, *, edist, prior_weight):
    knndf = knng.knn_df 
    symknn = knndf.assign(weight = kernel(knndf.distance, edist=edist))
    wmatrix = sp.coo_matrix((symknn.weight.values, (symknn.src_vertex.values, symknn.dst_vertex.values)))

    n = wmatrix.shape[0]
    diagw = sp.coo_matrix((np.ones(n)*prior_weight, (np.arange(n), np.arange(n))))
    wmatrix_tot = wmatrix + diagw
    norm_w = 1./np.array(wmatrix_tot.sum(axis=1)).reshape(-1)
    adj_matrix = wmatrix.tocsr()
    return adj_matrix, norm_w

def step(adj_mat, norm_w, prev_score, prior_weight, prior_score,  labels=None, label_weight=None):
    subtot = adj_mat @ prev_score + prior_weight*prior_score + label_weight*labels
    return subtot*norm_w

from sklearn.metrics import average_precision_score

def smoothen_scores(idx, term,  knndf, num_iters, prior_weight, **kwargs):
    qvec = idx.string2vec(term).reshape(-1)
    adj_mat, norm_w = prepare(knndf, prior_weight=prior_weight, **kwargs)
    logit_prior = idx.vectors @ qvec
    score_prior = sigmoid(logit_prior * 2)
    prev_scores = score_prior
    # ytrue = qgt[term].values 
    # ap = average_precision_score(ytrue, prev_scores)
    #print(ap)
    for i in range(num_iters): # do more than 1 step
        scores  =  step(adj_mat, norm_w, prev_score=prev_scores, 
                        prior_score=score_prior, prior_weight=prior_weight)
        # ap = average_precision_score(ytrue, scores)
        delta = np.sqrt(np.square(scores - prev_scores).mean())
        # print(f'{ap:.03f} {delta:.02e}')
        prev_scores = scores

    return scores


class LabelPropagationRanker:
    def __init__(self, knng : KNNGraph, init_scores=None, calib_a=2., calib_b=-1., prior_weight=1., kval=5, edist=.1, num_iters=2):
        self.knng : KNNGraph = knng

        if init_scores is None:
            self.prior_scores = np.ones(self.knng.nvecs)*.1 # base if nothing is given
        else:
            self.prior_scores = sigmoid(calib_a*init_scores + calib_b)

        self.prior_weight = prior_weight
        self.edist = edist
        self.kval = kval
        self.num_iters = num_iters

        self.adj_mat, self.norm_w = prepare(self.knng, prior_weight=prior_weight, edist=edist)
        
        self.labels = np.zeros_like(self.prior_scores)
        self.is_labeled = np.zeros_like(self.prior_scores)        
        self.all_indices = pr.FrozenBitMap(range(self.knng.nvecs))

        self._scores = self.prior_scores.copy()
        self._propagate(num_iters=self.num_iters)

    def _propagate(self, num_iters):
        labeled_prior = self.prior_scores * (1-self.is_labeled) + self.labels * self.is_labeled
        scores = self._scores # use previous scores as starting point
        for _ in range(num_iters): 
            prev_score = scores
            subtot = (self.adj_mat @ prev_score) + self.prior_weight*labeled_prior
            scores = subtot*self.norm_w

            ## override scores with labels 
            scores = scores * (1 - self.is_labeled) + self.labels * self.is_labeled
            norm = np.linalg.norm(prev_score - scores)
            # print(f'norm delta : {norm}')

        return scores
                
    def update(self, idx, label):
        idx = int(idx)
        label = float(label)
        
        assert np.isclose(label,0) or np.isclose(label,1)
                
        self.labels[idx] = label
        self.is_labeled[idx] = 1
                
        pscores = self._propagate(self.num_iters)
        self._scores = pscores

    def current_scores(self):
        return self._scores

    def top_k(self, k, unlabeled_only=True):
        if unlabeled_only:
            subset = np.where(self.is_labeled < 1)[0]
        else: 
            subset = np.array(self.all_indices)
            
        raw_scores = self.current_scores()
        
        topk_positions = np.argsort(-raw_scores[subset])[:k]
        topk_indices = subset[topk_positions]
        
        return topk_indices, raw_scores[topk_indices]
