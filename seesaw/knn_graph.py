from seesaw.services import parallel_read_parquet, get_parquet
import pandas as pd
import numpy as np
import pynndescent
import scipy.sparse as sp
import os


def rbf_kernel(edist):
    assert edist > 0
    """ edist is how far the distance has to grow for the weight to fall from max of 1. to 1/e """
    spread = 1./edist

    def kernel(arr):
        ## arr is given as cosine distance =  1 - cossim, ranging from 0 to 2
        assert arr.min() >= -.0001
        assert arr.max() <= 2.0001
        
        dist = arr.astype('float64')*spread
        return np.exp(-dist)

    return kernel

def knn_kernel(edist=2.1):
    assert edist > 0.
    """edist is distance beyond which we will discard points. 2 means not discarded"""
    def kernel(arr):
        return (arr <= edist).astype('float32')

    return kernel

def get_weight_matrix(df, kfun, self_edges=True, normalized=False) -> sp.coo_array:
    n = df.src_vertex.unique().shape[0]
    

    vertices = df[df.src_vertex == df.dst_vertex]
    assert vertices.shape[0] == n, f'{vertices.shape[0]=}, {n=}'
    # some edges in the df are k nn edges for both vertices, and both views will show up  in the df.
    # while some edges are only for one of the vertices. 
    # when adding the matrix below, those repeated edges will show up with a count of 2.
    adjacency_m = sp.coo_array( (np.ones(df.shape[0]), (df.src_vertex.values, df.dst_vertex.values)), shape=(n,n))
    symmetric_adj = adjacency_m.T + adjacency_m
    
    ## use metric as weight
    edge_weight_array = kfun(df.distance.values)
    assert (edge_weight_array > 0).all(), 'edge weights mut be positive'
    # edge weights must be positive bc zero values break sparse matrix rep. assumptions

    weight_mat =  sp.coo_array( (edge_weight_array, (df.src_vertex.values, df.dst_vertex.values)), shape=(n,n))
    symmetric_weight = weight_mat.T + weight_mat
    
    weight_values = symmetric_weight[symmetric_weight.nonzero()]
    edge_counts = symmetric_adj[symmetric_adj.nonzero()]
    adjusted_values = weight_values/edge_counts
    
    ## fix double counted values
    symmetric_weight[symmetric_weight.nonzero()] = adjusted_values.copy()
    out_w = symmetric_weight
    
    diag_iis = np.arange(n)
    
    assert np.isclose(kfun(np.zeros(1)), np.ones(1)) # sanity check on kfun

    ## assert diagonal is set to 1s
    ## this checks we are dealing ok with repeated edges ok
    isclose = np.isclose(out_w.diagonal(), 1., atol=1e-5)
    assert isclose.all()

    if not self_edges:
        out_w.setdiag(0.) # mutates out_w
    
    if normalized:
        ## D^-1/2 @ W @ D^-1/2
        Dvec = out_w.sum(axis=-1).reshape(-1)
        sqrt_invDmat = sp.coo_array( (1./np.sqrt(Dvec), (diag_iis, diag_iis)), shape=(n,n) )
        tmp = out_w.tocsr() @ sqrt_invDmat.tocsc() # type csr
        out_w = sqrt_invDmat.tocsr() @ tmp.tocsc()
            
    assert np.isclose(out_w.sum(axis=0), out_w.sum(axis=1)).all(), 'expect symmetric in any scenario'

    return out_w

def get_lookup_ranges(sorted_col, nvecs):
    cts = sorted_col.value_counts().sort_index()
    counts_filled = cts.reindex(np.arange(-1, nvecs), fill_value=0)
    ind_ptr = counts_filled.cumsum().values
    return ind_ptr

def post_process_graph_df(df, nvec):
    """ ensures graph has self edges, that edges are ranked, and that the datatypes are similar in all
    """
    ## make distances stop at 0, rather than sometimes cross slightly below
    ## check df column names and types
    for col in ['src_vertex', 'dst_vertex']:
        df = df.assign(**{col:df[col].astype('int32')})
    df = df.assign(distance=np.clip(df.distance.values.astype('float32'), a_min=0., a_max=None))


    df = df[df.src_vertex != df.dst_vertex] # filter out existing self-edges (they sometimes appear non deterministically)
    ranks = df.groupby('src_vertex').distance.rank('first').astype('int32') # rank starts at 1
    df = df.assign(dst_rank=ranks)

    ### re-add self edges to everything to every node so the number of vertices is always well defined from the 
    ## edges themselves after filtering to k nn.
    self_df = pd.DataFrame({'src_vertex':np.arange(nvec).astype('int32'), 
                            'dst_vertex':np.arange(nvec).astype('int32'),
                            'distance':np.zeros(nvec).astype('float32'),
                            'dst_rank':np.zeros(nvec).astype('int32'), 
                            # rank 0 neighbor to itself regardless of whether the dataset has duplicates 
                            # so we never lose diagonal
                           })
    
    df = pd.concat([df, self_df], ignore_index=True)
    df = df.sort_values(['src_vertex', 'dst_rank']).reset_index(drop=True)
    return df

def compute_exact_knn(vectors, n_neighbors):
    kmax = n_neighbors
    k = min(kmax + 1,vectors.shape[0])
    all_pairs = 1. - (vectors @ vectors.T)
    topk = np.argsort(all_pairs, axis=-1)

    dst_vertex = topk[:,:k]
    src_vertex,_ = np.indices(dimensions=dst_vertex.shape)

    src_vertex = src_vertex.reshape(-1)
    dst_vertex = dst_vertex.reshape(-1)
    distance = all_pairs[(src_vertex, dst_vertex)]

    # distances_sq = np.take_along_axis(all_pairs, dst_vertex, axis=-1)
    # assert(src_vertex.shape == distances_sq.shape)
    df = pd.DataFrame(dict(src_vertex=src_vertex.astype('int32'),
                           dst_vertex=dst_vertex.astype('int32'), 
                            distance=distance.astype('float32')))
    

    df = post_process_graph_df(df, nvec=vectors.shape[0])
    return df

def compute_knn_from_nndescent(vectors, *, n_neighbors, n_jobs=-1, low_memory=False, **kwargs):
    """ returns a graph and also the index """
    ## diversify prob: 1 is less accurate than 0. throws some edges away
    # multiplier: larger is better, note it multiplies vs n_neighbors. helps avoid getting stuck
    # nneighbors > 50 recommended for dot product accuracy.
    index2 = pynndescent.NNDescent(vectors, n_neighbors=n_neighbors+1, metric='dot', 
                                    diversify_prob=0., pruning_degree_multiplier=3.,
                                   n_jobs=n_jobs, low_memory=low_memory, **kwargs)
    positions, distances = index2.neighbor_graph
    
    nvec, nneigh = positions.shape
    iis, _ = np.indices(dimensions=(nvec, nneigh))
    df = pd.DataFrame({ 'src_vertex':iis.reshape(-1).astype('int32'),
                        'dst_vertex':positions.reshape(-1).astype('int32'), 
                        'distance':distances.reshape(-1).astype('float32'),
                    })

    df = post_process_graph_df(df, nvec)
    return df

def compute_inter_frame_knn_graph(knng, idx): 
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

def compute_intra_frame_knn(meta_df, max_k=4, max_d=.2):
    def _make_intra_frame_knn_single_frame(gp):
        dbidx = gp.dbidx.values.astype('int32')[0]
        df = compute_exact_knn(gp.vectors.to_numpy(), kmax=max_k)
        df = df.query(f'distance < {max_d}')
        df = reprocess_df(df)
        df = df.assign(src_vertex=gp.index.values[df.src_vertex.values].astype('int32'),
            dst_vertex=gp.index.values[df.dst_vertex.values].astype('int32'), 
            src_dbidx=dbidx, dst_dbidx=dbidx)
        return df

    final_df = meta_df.groupby('dbidx').apply(_make_intra_frame_knn_single_frame).reset_index(drop=True)
    return final_df


def adjust_intra_frame_knn(global_idx, final_df, idx):
    sdf = global_idx.vector_meta.reset_index()
    pairs = sdf.groupby('dbidx')['index'].min()
    dbidx2minidx_old = dict(zip(pairs.index.values, pairs.values))

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

import pyroaring as pr

class KNNGraph:
    def __init__(self, knn_df, nvecs=None):
        self.knn_df = knn_df

        ks = knn_df.groupby('src_vertex').dst_rank.max()
        self._ks = ks 
        self.k = ks.min()
        self.maxk = ks.median()
        self.nvecs = ks.shape[0]
        self.ind_ptr = get_lookup_ranges(knn_df.src_vertex, self.nvecs)

    def _check_rep(self):
        srcs = pr.BitMap(self.knn_df.src_vertex.unique())
        dsts = pr.BitMap(self.knn_df.dst_vertex.unique())
        assert srcs == dsts, 'self edges should guarantee this'
        assert self._ks.index.max() + 1 == len(srcs), 'self edges guarantee this'

    def restrict_k(self, *, k, ):
        if k < self.maxk:
            knn_df = self.knn_df.query(f'dst_rank < {k}').reset_index(drop=True)
            return KNNGraph(knn_df)
        elif k > self.maxk:
            assert False, f'can only do up to k={self.k} neighbors based on input df'
        else:
            return self
          
    @staticmethod
    def from_file(path):
        pref_path = f'{path}/forward.parquet'

        ## hack: use cache for large datasets sharing same knng, not for subsets 
        if path.find('subset') == -1:
            use_cache = True
        else:
            use_cache = False
        
        if use_cache:
            print('using cache', pref_path)
            df = get_parquet(pref_path)
        else:
            print('not using cache', pref_path)
            # also don't use parallelism in that case
            df = parallel_read_parquet(pref_path, parallelism=0)

        df = df.assign(distance=np.clip(df.distance.values, a_min=0, a_max=None).astype('float32'))
        # TODO: add some sanity checks?
        return KNNGraph(df)

    def rev_lookup(self, dst_vertex) -> pd.DataFrame:
        return self.knn_df.iloc[self.ind_ptr[dst_vertex]:self.ind_ptr[dst_vertex+1]]
