import pyroaring as pr
import numpy as np
from typing import Tuple

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


    def remaining_indices(self) -> pr.BitMap:
        return self.all_indices - self.seen_indices


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

from typing import Optional
class Result:
    value : float
    index : int
    pruned_fraction : Optional[float]
    
    def __init__(self, value, index, pruned_fraction = None):
        self.value = value
        self.index = index
        self.pruned_fraction = pruned_fraction


## how do we structure this so it returns a new object of the 
class ProbabilityModel:
    dataset : Dataset
    def __init__(self, dataset):
        self.dataset = dataset

    def condition(self, idx, y) -> 'ProbabilityModel':
        ''' returns new model
        '''
        raise NotImplementedError()

    def predict_proba(self, idx : np.ndarray ) -> np.ndarray:
        raise NotImplementedError()

    def top_k_remaining(self, top_k : int) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: if scores are identical (can be in some cases), break ties randomly.
        idxs = self.dataset.remaining_indices()
        curr_pred = self.predict_proba(idxs)
        desc_order = np.argsort(-curr_pred)
        ret_idxs = desc_order[:top_k]

        ans = []
        for idx in ret_idxs:
            ans.append(idxs[int(idx)])

        idx_ans = np.array(ans)
        return idx_ans, curr_pred[ret_idxs]

    def probability_bound(self, n) -> float:
        ''' upper bound on max p_i if we added n more positive results '''
        raise NotImplementedError
