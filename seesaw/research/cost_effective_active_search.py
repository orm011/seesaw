import pyroaring as pr
import numpy as np
from  seesaw.research.npb_distribution import NPBDistribution
import math

class Dataset:
    idx2label : dict
    vectors : np.ndarray
    all_indices : pr.FrozenBitMap
    seen_indices : pr.BitMap
    
    
    def __init__(self, idx2label, seen_indices, all_indices, vectors):
        self.idx2label = idx2label
        self.vectors = vectors
        self.seen_indices = seen_indices
        self.all_indices = all_indices
        
    @staticmethod
    def from_vectors(vectors):
        all_indices = pr.FrozenBitMap(range(len(vectors)))
        return Dataset({}, pr.BitMap(), all_indices, vectors)

    @staticmethod
    def from_labels(idxs, labels, vectors):
        idx2label = dict(zip(idxs, labels))
        all_indices = pr.FrozenBitMap(range(len(vectors)))
        return Dataset(idx2label, pr.BitMap(idxs), all_indices, vectors)

    
    def with_label(self, i, y):
        new_idx2label = self.idx2label.copy()
        new_idx2label[i] = y
        new_indices = self.seen_indices.copy()
        new_indices.add(i)
        
        assert i in self.all_indices        
        return Dataset(new_idx2label, new_indices, self.all_indices, self.vectors)
    
    def get_labels(self):
        idxs = np.array(self.seen_indices)
        labs = np.array([self.idx2label[idx] for idx in idxs])
        return idxs, labs


    def remaining_indices(self):
        return np.array(self.all_indices - self.seen_indices)


def test_dataset():
    arr2 = np.random.randn(10, 20)
    d0 = Dataset.from_vectors(arr2)
    
    d1 = d0.with_label(5, 1)    
    assert len(d1.idx2label) == 1
    assert d1.idx2label[5] == 1

    assert 5 not in d1.remaining_indices()
    
    d2 = d0.with_label(6, 0)
    assert len(d2.idx2label) == 1
    assert d2.idx2label[6] == 0
    assert  6 not in d2.remaining_indices()

    assert 5 in d2.remaining_indices()
    assert 6 in d1.remaining_indices()

    assert len(d0.idx2label) == 0
    assert 5 in d0.remaining_indices()
    assert 6 in d0.remaining_indices()
    
    
    d3 = d1.with_label(1, 0)
    assert len(d3.seen_indices) == 2
    assert d3.idx2label[1] == 0
    assert d3.idx2label[5] == 1
    assert 1 not in d3.remaining_indices()
    assert 5 not in d3.remaining_indices()

import torch

class Result:
    expected_cost : float
    index : int
    
    def __init__(self, expected_cost, index):
        self.expected_cost = expected_cost
        self.index = index


## how do we structure this so it returns a new object of the 
class IncrementalModel:
    dataset : Dataset
    def __init__(self, dataset):
        self.dataset = dataset

    def with_label(self, idx, y) -> 'IncrementalModel':
        ''' returns new model
        '''
        raise NotImplementedError()

    def predict_proba(self, idx : np.ndarray ) -> np.ndarray:
        raise NotImplementedError()

def expected_cost(idx, *, r : int,  t : int,  model : IncrementalModel) -> float:
    p = model.predict_proba(np.array([idx])).item()
    # case y = 1
    res1 = min_expected_cost_approx(r-1,  t=t-1, model = model.with_label(idx,1))

    # case y = 0
    res0 = min_expected_cost_approx(r,  t=t-1, model = model.with_label(idx,0))
    
    return p*res1.expected_cost + (1-p)*res0.expected_cost

def min_expected_cost_approx(r : int, *,  top_k : int = None, t : int, model : IncrementalModel) -> Result:
    if t == 0:
        indices = model.dataset.remaining_indices()
        probs = model.predict_proba(indices)
        probs = torch.from_numpy(probs)
        desc_order = torch.argsort(-probs)
        desc_probs = probs[desc_order]        
        cost =  NPBDistribution(r, desc_probs).expectation(method='accu_prime')
        index = indices[desc_order[0]]
        return Result(expected_cost=cost, index=index)

    min_idx = None
    min_cost = math.inf

    idxs = model.dataset.remaining_indices()
    curr_pred = model.predict_proba(idxs)
    desc_order = np.argsort(-curr_pred)
    top_k_idxs = idxs[desc_order[:top_k]]

    for idx in top_k_idxs:
        c = expected_cost(idx, r=r, t=t, model=model)
        if c < min_cost:
            min_idx = idx
            min_cost = c
            
    return Result(expected_cost=min_cost, index=min_idx)