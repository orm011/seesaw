from seesaw.services import get_parquet
import pandas as pd
import numpy as np
import pynndescent
import scipy.sparse as sp


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

def get_weight_matrix(df, *, kfun, self_edges=False, normalized, laplacian=False, symmetric=True) -> sp.csr_array:
    assert not self_edges

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
    assert (edge_weight_array >= 0).all(), 'edge weights mut be non-negative'

    # some distances become 0 after using kernel func.
    mask = edge_weight_array > 0
    # edge weights must be positive bc zero values break sparse matrix rep. assumptions
    weight_mat =  sp.coo_array( (edge_weight_array[mask], 
                        (df.src_vertex.values[mask], df.dst_vertex.values[mask])), shape=(n,n))

    if symmetric:
        symmetric_weight = weight_mat.T + weight_mat        
        pos = symmetric_adj.nonzero()
        weight_values = symmetric_weight[pos]
        edge_counts = symmetric_adj[pos]
        adjusted_values = weight_values/edge_counts
        symmetric_weight[pos] = adjusted_values
        out_w = symmetric_weight
        diag_iis = np.arange(n)

        ## assert diagonal is set to 1s
        ## this checks we are dealing ok with repeated edges ok
        isclose = np.isclose(out_w.diagonal(), 1., atol=1e-5)
        assert isclose.all()
        assert np.isclose(kfun(np.zeros(1)), np.ones(1)) # sanity check on kfun
    else:
        out_w = weight_mat

    out_w.setdiag(0.) # mutates out_w

    assert np.isclose(out_w.diagonal(), 0., atol=1e-5).all()
    
    D = out_w.sum(axis=0)
    assert (D > 0).all(), 'no zero degree nodes allowed'
    
    if laplacian:
        assert symmetric
        assert not self_edges, 'unknown meaning of this parameter combination'

        out_w = -out_w
        out_w.setdiag(D)
        assert np.isclose(out_w.sum(0) , 0).all()
        assert np.isclose(out_w.diagonal(), D).all()

        if normalized:
            ## D^-1/2 @ L @ D^-1/2
            sqrt_inv_Dmat = sp.coo_array( (1./np.sqrt(D.reshape(-1)), (diag_iis, diag_iis)), shape=(n,n) )
            out_w = sqrt_inv_Dmat @ (out_w @ sqrt_inv_Dmat) 

    out_w = out_w.tocsr()
    out_w.sum_duplicates()
    out_w.sort_indices()
    assert out_w.has_sorted_indices    

    if symmetric:
        assert np.isclose(out_w.sum(axis=0), out_w.sum(axis=1)).all(), 'expect symmetric in any scenario'

    

    return out_w
    
def edge_loss(laplacian_m, labels):
    return labels @ (laplacian_m @ labels)

def test_simple_edge_loss():
    simple_edge = pd.DataFrame({'src_vertex':[0, 0, 1, 1,], 'dst_vertex':[0,1,1,0], 'distance':[0., 1., 0., 1.], 'dst_rank':[0,1,0,1]})    
    test_knng = KNNGraph(simple_edge)
    
    laplacian_m = get_weight_matrix(test_knng.knn_df, kfun=rbf_kernel(10000.), normalized=False, self_edges=False, laplacian=True)
    
    l00 = edge_loss(laplacian_m, np.array([0,0]))
    l11 = edge_loss(laplacian_m, np.array([1,1]))
    l01 = edge_loss(laplacian_m, np.array([0,1]))
    l10 = edge_loss(laplacian_m, np.array([1,0]))
    
    assert np.isclose(l00,0)
    assert np.isclose(l11,0)
    assert np.abs(l01 - 1.) < .001
    assert np.isclose(l10,l01)
    
    laplacian_m2 = get_weight_matrix(test_knng.knn_df, kfun=rbf_kernel(.0001), normalized=False, self_edges=False, laplacian=True)
    l00 = edge_loss(laplacian_m2, np.array([0,0]))
    l11 = edge_loss(laplacian_m2, np.array([1,1]))
    l01 = edge_loss(laplacian_m2, np.array([0,1]))
    l10 = edge_loss(laplacian_m2, np.array([1,0]))
    
    assert np.isclose(l00,0)
    assert np.isclose(l11,0)
    assert np.abs(l01 - 0) < .001
    assert np.isclose(l10,l01)

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
                                    diversify_prob=0., pruning_degree_multiplier=4.,
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

def factor_neighbors(knng, idx, k_intra):
    ''' returns a new df with neighbors of different dbidxs having a separate rank
        than neighbors from the same dbidx. choosing rank < 4 will pick the closest 4 other frames,
        as well as vectors within the frame.
    '''
    dbidxs = idx.vector_meta.dbidx.astype('int32').values
    df = knng.knn_df
    df = df.assign(src_dbidx=dbidxs[df.src_vertex.values], 
              dst_dbidx=dbidxs[df.dst_vertex.values])
    
    def _make_inter(df, k):
        df = df.query('src_dbidx != dst_dbidx')
        edge_ranks = df.groupby(['src_vertex', 'dst_dbidx']).distance.rank('first').astype('int')
        df = df.assign(edge_rank=edge_ranks)
        divdf = df[df.edge_rank <= k]
        new_ranks = divdf.groupby(['src_vertex']).distance.rank('first').sub(1).astype('int')
        divdf = divdf.assign(dst_rank=new_ranks).drop('edge_rank', axis=1)
        return divdf
    
    def _make_intra(df, k):
        df = df.query('src_dbidx == dst_dbidx')
        rank_within_frame = df.groupby('src_vertex').distance.rank('first').astype('int')
        df = df.assign(dst_rank=rank_within_frame)
        df = df.query(f'dst_rank <= {k}')
        return df

    inter = _make_inter(df, k=1) # 1 per dbidx
    intra = _make_intra(df, k=k_intra) # more within single frame
    both = pd.concat([inter, intra], ignore_index=True)
    return both

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
            cache = True
        else:
            cache = False

        df = get_parquet(pref_path, parallelism=0, cache=cache)
        return KNNGraph(df)

    def rev_lookup(self, dst_vertex) -> pd.DataFrame:
        return self.knn_df.iloc[self.ind_ptr[dst_vertex]:self.ind_ptr[dst_vertex+1]]
