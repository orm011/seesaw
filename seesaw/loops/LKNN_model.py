    
from ..research.active_search.common import ProbabilityModel, Dataset
import numpy as np
import scipy.sparse as sp
from typing import Tuple
import pyroaring as pr
import math

class LazyTopK:
    def __init__(self, dataset : Dataset, desc_idxs, desc_scores, desc_changed_idxs, desc_changed_scores):
        self.dataset : Dataset = dataset

        self.desc_scores = desc_scores
        self.desc_idxs = desc_idxs

        self.changed_idxs = desc_changed_idxs
        self.changed_idx_set= pr.BitMap(desc_changed_idxs)
        self.changed_scores = desc_changed_scores

    def __iter__(self):
        i = 0
        j = 0

        while True:
            while i < len(self.desc_scores) and (
                    self.desc_idxs[i] in self.dataset.seen_indices
                    or 
                    self.desc_idxs[i] in self.changed_idx_set
                 ):
                i+=1

            if i < len(self.desc_scores):
                top_orig_score = self.desc_scores[i]
            else:
                top_orig_score = - math.inf

            while j < len(self.changed_idxs) and (
                    self.changed_idxs[j] in self.dataset.seen_indices
                ):
                j+=1
            if j < len(self.changed_idxs):
                top_changed_score = self.changed_scores[j]
            else:
                top_changed_score = - math.inf

            if i == len(self.desc_scores) and j == len(self.changed_scores):
                break

            if top_orig_score >= top_changed_score:
                yield (self.desc_idxs[i], top_orig_score)
                i+=1
            else:
                yield (self.changed_idxs[j], top_changed_score)
                j+=1

    def top_k_remaining(self, k):
        vals = []
        idxs = []
        for i, (idx, val) in enumerate(iter(self)):
            if i >= k:
                break

            idxs.append(idx)
            vals.append(val)

        return np.array(idxs), np.array(vals)

class LKNNModel(ProbabilityModel):
    ''' Implements L-KNN prob. model used in Active Search paper.
    '''    
    def __init__(self, dataset : Dataset, gamma : float, matrix : sp.csr_array, numerators : np.ndarray, denominators : np.ndarray, lz_topk: LazyTopK):

        super().__init__(dataset)
        self.matrix = matrix
        self.numerators = numerators
        self.denominators = denominators
        self.gamma = gamma
        self.lz_topk = lz_topk

        assert dataset.vectors.shape[0] == matrix.shape[0]

        ## set probs to estimates, then replace estimates with labels
        self._probs = (gamma + numerators) / (1 + denominators)

    @staticmethod
    def from_dataset( dataset : Dataset, weight_matrix : sp.csr_array, gamma : float):
        assert weight_matrix.format == 'csr'
        assert len(dataset.idx2label) == 0, 'not implemented other case'
        ## need to initialize numerator and denominator
        sz = weight_matrix.shape[0]
        numerators=np.zeros(sz)
        denominators=np.zeros(sz)
        init_scores = (numerators + gamma) / (1 + denominators)
        init_sort = np.argsort(-init_scores)

        lz_topk = LazyTopK(dataset=dataset, desc_idxs=init_sort, desc_scores=init_scores[init_sort], 
                            desc_changed_scores=np.array([]),desc_changed_idxs=np.array([]))

        return LKNNModel(dataset, gamma=gamma, matrix=weight_matrix, numerators=np.zeros(sz), denominators=np.zeros(sz), lz_topk=lz_topk)


    def condition(self, idx, y) -> 'LKNNModel':
        ''' returns new model
        '''

        numerators = self.numerators.copy()
        denominators = self.denominators.copy()


        row  = self.matrix[[idx],:] # may include itself, but will ignore these
        _, neighbors = row.nonzero()

        curr_label = self.dataset.idx2label.get(idx, None)
        if curr_label is None:
            numerators[neighbors] += y
            denominators[neighbors] += 1
        elif curr_label != y:
            numerators[neighbors] += (y - curr_label)
        else: # do nothing.
            pass


        new_scores = (numerators[neighbors] + self.gamma)/(1 + denominators[neighbors])
        # only idx and neighbors may have changed
        # new order = merge remaining order desc and neighbors
        new_dataset = self.dataset.with_label(idx, y)

        ## new changed_idxs = old changed_idxs updated with new
        old_dict = dict(zip(self.lz_topk.changed_idxs, self.lz_topk.changed_scores))
        new_dict = dict(zip(neighbors, new_scores))

        old_dict.update(new_dict)
        idxs = np.array(list(old_dict.keys()))
        values = np.array(list(old_dict.values()))

        desc_order = np.argsort(-values)
        desc_changed_idxs = idxs[desc_order]
        desc_changed_scores = values[desc_order]
        
        lz_topk  =  LazyTopK(dataset=new_dataset, desc_idxs=self.lz_topk.desc_scores, desc_scores=self.lz_topk.desc_scores,
        desc_changed_scores=desc_changed_scores, desc_changed_idxs=desc_changed_idxs)

        return LKNNModel(new_dataset, gamma=self.gamma, matrix=self.matrix, numerators=numerators, denominators=denominators, lz_topk=lz_topk)

    def predict_proba(self, idxs : np.ndarray ) -> np.ndarray:
        return self._probs[idxs]

    def top_k_remaining(self, top_k : int) -> Tuple[np.ndarray, np.ndarray]:
        ## cheap form of finding the top k highest scoring without materializing all scores
        return self.lz_topk.top_k_remaining(k=top_k)

    def probability_bound(self, n) -> np.ndarray:
        idxs = self.dataset.remaining_indices()
        prob_bounds = (self.gamma + n + self.numerators[idxs])/(1 + n + self.denominators[idxs])
        return np.max(prob_bounds)