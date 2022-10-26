from seesaw.services import parallel_read_parquet
import pandas as pd
import numpy as np
import pynndescent
import scipy.sparse as sp
import os

def kernel(cosine_distance, edist):
    # edist = cosine distance needed to reduce neighbor weight to 1/e.
    # for cosine sim, the similarity ranges from -1 to 1. 
    spread = 1./edist
    return np.exp(-cosine_distance*spread)


def get_weight_matrix(knng, kfun, self_edges=True, normalized=False) -> sp.coo_array:
    df = knng.knn_df.query('is_forward') # remove pure reverse edges, we will add them below
    n = knng.nvecs
    diag_iis = np.arange(n)

    ## use metric as weight
    weights = kfun(df.distance)/(df.is_forward.astype('float') + df.is_reverse.astype('float'))
    ## division accounts for those edges that appear twice

    out_w = sp.coo_array( (weights, (df.src_vertex.values, df.dst_vertex.values)), shape=(n,n))
    out_w = (out_w + out_w.T) # making symmetric
    
    if self_edges:
        ## knng does not include self-edges. add self-edges
        self_weight = kfun(np.zeros(1))
        self_w = sp.coo_array( (np.ones(n) * self_weight, (diag_iis, diag_iis)), shape=(n,n))
        out_w = self_w + out_w


    if normalized:
        Dvec = out_w.sum(axis=-1).reshape(-1)
        sqrt_Dvec = np.sqrt(Dvec)
        sqrt_Dmat = sp.coo_array( (sqrt_Dvec, (diag_iis, diag_iis)), shape=(n,n) )

        tmp = out_w.tocsr() @ sqrt_Dmat.tocsc() # type csr
        out_w = sqrt_Dmat.tocsr() @ tmp.tocsc()

    assert np.isclose(out_w.sum(axis=0), out_w.sum(axis=1)).all(), 'expect symmetric'
    return out_w




def get_lookup_ranges(sorted_col, nvecs):
    cts = sorted_col.value_counts().sort_index()
    counts_filled = cts.reindex(np.arange(-1, nvecs), fill_value=0)
    ind_ptr = counts_filled.cumsum().values
    return ind_ptr



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
        # identity = (positions == np.arange(positions.shape[0]).reshape(-1,1))
        # any_identity = identity.sum(axis=1) > 0
        # exclude = identity
        # exclude[~any_identity, -1] = 1 # if there is no identity in the top k+1, exclude the k+1
        # assert (exclude.sum(axis=1) == 1).all()
        # positions1 = positions[~exclude].reshape(-1,n_neighbors)
        # distances1 = distances[~exclude].reshape(-1,n_neighbors)
        
        nvec, nneigh = positions.shape
        iis, _ = np.indices(dimensions=(nvec, nneigh))
        df = pd.DataFrame({ 'src_vertex':iis.reshape(-1).astype('int32'),
                            'dst_vertex':positions.reshape(-1).astype('int32'), 
                            'distance':distances.reshape(-1).astype('float32'),
                        })

        # print('postprocessing df')
        # knn_df = reprocess_df(knn_df)
        df[df.src_vertex != df.dst_vertex] # filter out self-edges (they appear non deterministically)
        # compute rank after filtering out self-edges
        ranks = df.groupby('src_vertex').distance.rank('first').sub(1).astype('int32')

        # some edges appear in both lists already, some do not.
        df = df.assign(dst_rank=ranks)

        knn_df.groupby('src_vertex').distance.rank('first').

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

        ## hack: use cache for large datasets sharing same knng, not for subsets 
        if path.find('subset') == -1:
            use_cache = True
        else:
            use_cache = False
        
        if os.path.exists(pref_path):
            if use_cache:
                print('using cache', pref_path)
                df = get_parquet(pref_path)
            else:
                print('not using cache', pref_path)
                df = parallel_read_parquet(pref_path, parallelism=parallelism)
            nvecs = df.src_vertex.max() + 1
            # for some reason dst_rank is wrong sometimes. just recompute it
            df = df.assign(dst_rank=(df.groupby('src_vertex').distance.rank('first') - 1).astype('int32'))
            return KNNGraph(df, nvecs)
        else:
            print('no sym.parquet found, computing')
            knn_df = parallel_read_parquet(f'{path}/forward.parquet', parallelism=parallelism)
            knn_df = reprocess_df(knn_df)
            nvecs = knn_df.src_vertex.max() + 1
            graph = KNNGraph(knn_df, nvecs)
            graph.save(path, num_blocks=1)
            return KNNGraph.from_file(path, parallelism=parallelism)


    def rev_lookup(self, dst_vertex) -> pd.DataFrame:
        return self.knn_df.iloc[self.ind_ptr[dst_vertex]:self.ind_ptr[dst_vertex+1]]
