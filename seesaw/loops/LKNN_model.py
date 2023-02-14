    
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
    def __init__(self, dataset : Dataset, gamma : float, matrix : sp.csr_array, numerators : np.ndarray, denominators : np.ndarray):

        super().__init__(dataset)
        self.matrix = matrix
        self.numerators = numerators
        self.denominators = denominators
        self.gamma = gamma

        assert dataset.vectors.shape[0] == matrix.shape[0]
        # print(f'{matrix.shape=}')

        ## set probs to estimates, then replace estimates with labels
        self._probs = (gamma + numerators) / (1 + denominators)
#        self.order_desc = order_desc
#        assert order_desc.shape[0] == self._probs.shape[0]

        if len(dataset.seen_indices) > 0:
            idxs, labels = dataset.get_labels()
            self._probs[idxs] = labels


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

        return LKNNModel(dataset, gamma=gamma, matrix=weight_matrix, numerators=np.zeros(sz), denominators=np.zeros(sz))


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


        # only idx and neighbors may have changed
        # new order = merge remaining order desc and neighbors


        new_dataset = self.dataset.with_label(idx, y)


        return LKNNModel(new_dataset, gamma=self.gamma, matrix=self.matrix, numerators=numerators, denominators=denominators)

    def predict_proba(self, idxs : np.ndarray ) -> np.ndarray:
        return self._probs[idxs]

    def top_k_remaining(self, top_k : int) -> Tuple[np.ndarray, np.ndarray]:
        ## cheap form of finding the top k highest scoring without materializing all scores
        pass

    def probability_bound(self, n) -> np.ndarray:
        idxs = self.dataset.remaining_indices()
        prob_bounds = (self.gamma + n + self.numerators[idxs])/(1 + n + self.denominators[idxs])
        return np.max(prob_bounds)