from seesaw.services import get_parquet
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


def get_lookup_ranges(sorted_col, nvecs):
    cts = sorted_col.value_counts().sort_index()
    counts_filled = cts.reindex(np.arange(-1, nvecs), fill_value=0)
    ind_ptr = counts_filled.cumsum().values
    return ind_ptr

def reprocess_df(df0):
    df = df0.assign(src_vertex=df0.src_vertex.astype('int32'), 
                    dst_vertex=df0.dst_vertex.astype('int32'),
                    distance=df0.distance.astype('float32'),
                    dst_rank=df0.dst_rank.astype('int32'))
    
    df_rev = df.assign(src_vertex=df.dst_vertex, dst_vertex=df.src_vertex)
    df_all = pd.concat([df.assign(is_forward=True, is_reverse=False), 
                        df_rev.assign(is_forward=False, is_reverse=True)], ignore_index=True)

    df_all = df_all.sort_values(['src_vertex'])
    dups = df_all.duplicated(['src_vertex', 'dst_vertex'], keep=False)
    df_all = df_all.assign(is_forward=(df_all.is_forward | dups),
                     is_reverse=(df_all.is_reverse | dups))
    df_all = df_all.drop_duplicates(['src_vertex', 'dst_vertex']).reset_index(drop=True)
    return df_all

def uniquify_knn_graph(knng, idx): 
    dbidxs = idx.vector_meta.dbidx.astype('int32').values
    df = knng.knn_df
    df = df.assign(src_dbidx=dbidxs[df.src_vertex.values], 
              dst_dbidx=dbidxs[df.dst_vertex.values])

    df = df[df.src_dbidx != df.dst_dbidx]
    per_dest = df.groupby(['src_vertex','dst_dbidx']).distance.idxmin()
    pddf = df.loc[per_dest.values]

    pddf = pddf.assign(dst_rank=pddf.groupby(['src_vertex']).distance.rank('first').astype('int32'))
    pddf = pddf.reset_index(drop=True)
    return pddf

class KNNGraph:
    def __init__(self, knn_df, nvecs):
        self.knn_df = knn_df
        self.nvecs = nvecs
        self.k = knn_df.dst_rank.max() + 1        
        self.ind_ptr = get_lookup_ranges(knn_df.src_vertex, self.nvecs)

    def restrict_k(self, *, k):
        if k < self.k:
            knn_df = self.knn_df.query(f'dst_rank < {k}').reset_index(drop=True)
            return KNNGraph(knn_df, self.nvecs)
        elif k > self.k:
            assert False, f'can only do up to k={self.k} neighbors based on input df'
        else:
            return self

    def forward_graph(self):
        knn_df = self.knn_df
        return KNNGraph(knn_df[knn_df.is_forward], self.nvecs)
    
    def reverse_graph(self):
        knn_df = self.knn_df
        return KNNGraph(knn_df[knn_df.is_reverse], self.nvecs)

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
        positions1 = positions[~exclude].reshape(-1,n_neighbors)
        distances1 = distances[~exclude].reshape(-1,n_neighbors)

        nvec, nneigh = positions1.shape
        iis, jjs = np.meshgrid(np.arange(nvec), np.arange(nneigh), indexing='ij')

        knn_df = pd.DataFrame({ 'src_vertex':iis.reshape(-1).astype('int32'),
                                'dst_vertex':positions1.reshape(-1).astype('int32'), 
                                'distance':distances1.reshape(-1).astype('float32'),
                                'dst_rank':jjs.reshape(-1).astype('int32'),
                            })

        print('postprocessing df')
        knn_df = reprocess_df(knn_df)
        knn_graph = KNNGraph(knn_df, nvecs=vectors.shape[0])
        return knn_graph, index2

                
    def save(self, path, num_blocks=10, overwrite=False):
        import ray.data
        import shutil

        if os.path.exists(path) and overwrite:
            shutil.rmtree(path)

        os.makedirs(path, exist_ok=True)

        if num_blocks > 1:
            ds = ray.data.from_pandas(self.knn_df).lazy()
            ds.repartition(num_blocks=num_blocks).write_parquet(f'{path}/sym.parquet')
        else:
            self.knn_df.to_parquet(f'{path}/sym.parquet')            
    
    @staticmethod
    def from_file(path, parallelism=0):

        pref_path = f'{path}/sym.parquet'
        if os.path.exists(pref_path):
            df = parallel_read_parquet(pref_path, parallelism=parallelism)
            nvecs = df.src_vertex.max() + 1
            return KNNGraph(df, nvecs)

        else:
            print('no sym.parquet found, computing')
            knn_df = parallel_read_parquet(f'{path}/forward.parquet', parallelism=parallelism)
            knn_df = reprocess_df(knn_df)
            nvecs = knn_df.src_vertex.max() + 1
            graph = KNNGraph(knn_df, nvecs)
            graph.save(path, num_blocks=1)
            return graph


    def rev_lookup(self, dst_vertex) -> pd.DataFrame:
        return self.knn_df.iloc[self.ind_ptr[dst_vertex]:self.ind_ptr[dst_vertex+1]]


def compute_exact_knn(vectors, kmax):
    k = min(kmax + 1,vectors.shape[0])
    all_pairs = 1. - (vectors @ vectors.T)
    topk = np.argsort(all_pairs, axis=-1)
    n = all_pairs.shape[0]
    dst_vertex = topk[:,:k]
    src_vertex = np.repeat(np.arange(n).reshape(-1,1), repeats=k, axis=1)

    assert(dst_vertex.shape == src_vertex.shape)

    distances_sq = np.take_along_axis(all_pairs, dst_vertex, axis=-1)
    assert(src_vertex.shape == distances_sq.shape)
    df1 = pd.DataFrame(dict(src_vertex=src_vertex.reshape(-1).astype('int32'), dst_vertex=dst_vertex.reshape(-1).astype('int32'), 
                  distance=distances_sq.reshape(-1).astype('float32')))
    
    ## remove same vertex
    df1 = df1[df1.src_vertex != df1.dst_vertex]
    
    ## add dst rank
    df1 = df1.assign(dst_rank=df1.groupby(['src_vertex']).distance.rank('first').sub(1).astype('int32'))
    
    ## filter any extra neighbors
    df1 = df1[df1.dst_rank < kmax]
    return df1

#     all_meta = idx_top.vector_meta.reset_index(drop=False)
#    pairs = all_meta.groupby('dbidx')['index'].min()
#    dbidx2minidx_old = dict(zip(pairs.index.values, pairs.values))
#


def make_intra_frame_knn(dbidx2minidx_old, final_df, idx):
    sdf = idx.vector_meta.reset_index(drop=False)

    
    pairs = sdf.groupby('dbidx')['index'].min()
    dbidx2minidx_new = dict(zip(pairs.index.values, pairs.values))

    def per_group(gp):
        k = int(gp.dst_dbidx.iloc[0])
        delta = dbidx2minidx_new[k] - dbidx2minidx_old[k]
        gp = gp.eval(f'src_vertex = src_vertex + {delta}\n dst_vertex=dst_vertex + {delta}')
        return gp

    knn_df_frame = final_df[final_df.src_dbidx.isin(set(dbidx2minidx_new.keys()))]
    remapped = knn_df_frame.groupby('src_dbidx', group_keys=False).apply(per_group)
    return remapped

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

    def update(self, idxs, labels):
        for idx, label in zip(idxs, labels):
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
    n = knng.nvecs

    wmatrix = sp.coo_matrix( (symknn.weight.values, (symknn.src_vertex.values, symknn.dst_vertex.values)), shape=(n, n))
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
    def __init__(self, knng : KNNGraph, init_scores=None, calib_a=2., calib_b=-1., prior_weight=1., kval=5, edist=.1, num_iters=2, **other):
        self.knng : KNNGraph = knng

        self.calib_a = calib_a
        self.calib_b = calib_b
        self.prior_weight = prior_weight
        self.edist = edist
        self.kval = kval
        self.num_iters = num_iters

        self.is_labeled = np.zeros(self.knng.nvecs)
        self.labels = np.zeros(self.knng.nvecs)

        self.all_indices = pr.FrozenBitMap(range(self.knng.nvecs))
        self.adj_mat, self.norm_w = prepare(self.knng, prior_weight=prior_weight, edist=edist)

        if init_scores is not None:
            self.set_base_scores(init_scores)


    def set_base_scores(self, init_scores):
        assert self.knng.nvecs == init_scores.shape[0]
        self.prior_scores = sigmoid(self.calib_a*init_scores + self.calib_b)
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
                
    def update(self, idxs, labels):

        for idx, label in zip(idxs, labels):
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
