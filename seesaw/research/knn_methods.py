import numpy as np
import pyroaring as pr
from ray.data.extensions import TensorArray
import pynndescent 
import ray.data
import pandas as pd
from ..services import get_parquet


def extract_knn_ds(df, n=30):
    index2 = pynndescent.NNDescent(df.vectors.to_numpy(), n_neighbors=n+1, metric='dot')
    positions, distances = index2.neighbor_graph
    identity = (positions == np.arange(positions.shape[0]).reshape(-1,1))
    
    positions1 = positions[~identity].reshape(-1,30)
    distances1 = distances[~identity].reshape(-1,30)
    knn_df = pd.DataFrame({'positions':TensorArray(positions1), 'distances':TensorArray(distances1)})
    return knn_df

def save_knn_df(knn_df, index_root):
    knn_ds = ray.data.from_pandas(knn_df)
    knn_ds = knn_ds.repartition(50)
    knn_ds.write_parquet(f'{index_root}/knn_ds.parquet')

def load_knn_df(index_root):
    return get_parquet(f'{index_root}/knn_ds.parquet')

def getRKNN(knndf):
    posn = knndf['positions'].to_numpy()
    dist = knndf['distances'].to_numpy()
    indices = np.arange(posn.shape[0])
    indices = indices.reshape(-1,1).repeat(posn.shape[1], axis=1)
    
    idf = pd.DataFrame({'dest':posn.reshape(-1), 'orig':indices.reshape(-1), 'dist':dist.reshape(-1)})
    rkNN = idf.sort_values(['dest', 'dist']).reset_index(drop=True)
    return rkNN

from collections import defaultdict
def make_rknn_lookup(rkNN):
    rev_edge_list = defaultdict(lambda : np.array([], dtype=np.int64))
    for d,grp in rkNN.groupby('dest'):
        rev_edge_list[d] = grp['orig'].values
    return rev_edge_list


class SimpleKNNRanker:
    def __init__(self, knn_df, init_scores):
        nnindices = knn_df['positions'].to_numpy()
        assert nnindices.shape[0] == init_scores.shape[0]
        self.nnindices = nnindices

        rknndf= getRKNN(knn_df)
        self.rev_indices = make_rknn_lookup(rknndf)

        self.init_numerators = (init_scores + 1.)/2. # make them be between 0 and 1 just like labels, 
        self.init_denominators = np.ones_like(init_scores)
        
        self.numerators = np.zeros_like(self.init_numerators)
        self.denominators = np.zeros_like(self.init_denominators)
        
        self.is_labeled = np.zeros_like(self.init_numerators)
        
        self.all_indices = pr.FrozenBitMap(range(nnindices.shape[0]))
        
    def current_scores(self):
        mask = 1. - self.is_labeled
        num = self.init_numerators + mask*self.numerators
        denom = self.init_denominators + mask*self.denominators
        return num/denom
        
    def update(self, idx, label):
        idx = int(idx)
        label = float(label)
        
        assert np.isclose(label,0) or np.isclose(label,1)
        
        if self.is_labeled[idx] > 0: # if new label for old 
            old_label = self.init_numerators[idx]
            delta_denom = 0
            delta_num = label - old_label # erase old label and add new label
        else:
            delta_num = label
            delta_denom = 1
        
        self.init_numerators[idx] = label
        self.init_denominators[idx] = 1.
        self.is_labeled[idx] = 1.
                
        ## update scores for all v such that idx \in knn(v)
        rev_neighbors = self.rev_indices[idx]
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