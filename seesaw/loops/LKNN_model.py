    
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
        
        assert self.desc_scores.shape[0] == self.desc_idxs.shape[0]

        self.changed_idxs = desc_changed_idxs
        self.changed_idx_set= pr.BitMap(desc_changed_idxs.reshape(-1))
        self.changed_scores = desc_changed_scores
        assert self.changed_idxs.shape[0] == self.changed_scores.shape[0]
        assert len(self.changed_idx_set) == self.changed_scores.shape[0]

    def _iter_desc_scores(self):
        ignore_set = self.dataset.seen_indices.union(self.changed_idx_set)
        for (idx, score) in zip(self.desc_idxs, self.desc_scores):
            if idx not in ignore_set:
                yield (idx, score)
            else:
                pass

    def _iter_changed_scores(self):
        for (idx, score) in zip(self.changed_idxs, self.changed_scores):
            if idx not in self.dataset.seen_indices:
                yield (idx, score)
        
    def iter_desc(self):
        """ merges both streams in descending order, excluding any already seen elements
        """
        iter1 = self._iter_desc_scores()
        iter2 = self._iter_changed_scores()

        idx1, score1 = next(iter1)
        idx2, score2 = next(iter2)
        while (idx1 > -1) or (idx2 > -1):
            if score1 >= score2:
                yield (idx1, score1)
                idx1, score1 = next(iter1, (-1, -math.inf))
            else:
                yield (idx2, score2)
                idx2, score2 = next(iter2, (-1, -math.inf))

        assert (idx1 == -1) and (idx2 == -1), 'how did we get here'

    def top_k_remaining(self, k):
        vals = []
        idxs = []
        
        for i, (idx, val) in enumerate(self.iter_desc()):
            if i >= k:
                break

            idxs.append(idx)
            vals.append(val)

        return np.array(idxs), np.array(vals)

import numexpr
import numpy.random

def initial_gamma_array(gamma, shape):
    rnd = np.random.default_rng(seed=0)
    return rnd.normal(loc=gamma, scale=1e-6, size=shape)


class LKNNModel(ProbabilityModel):
    ''' Implements L-KNN prob. model used in Active Search paper.
    '''    
    def __init__(self, dataset : Dataset, gamma : np.ndarray, matrix : sp.csr_array, numerators : np.ndarray, denominators : np.ndarray, 
    score: np.ndarray, desc_idx : np.ndarray, desc_score : np.ndarray, desc_changed_idx : np.ndarray, desc_changed_score : np.ndarray):

        super().__init__(dataset)
        self.matrix = matrix
        self.numerators = numerators
        self.denominators = denominators
        self.score = score
        self.gamma = gamma
        assert (gamma.shape == self.numerators.shape)
        assert ((0 < gamma) & (gamma < 1)).all(), 'this could fail by chance, decrase var. or fix properly by applying sigmoid'

        
        self.desc_idx = desc_idx
        self.desc_score = desc_score

        ## starts off null, gets filled after first branching
        self.desc_changed_idx = desc_changed_idx
        self.desc_changed_score = desc_changed_score

        self._init_sets()
        
    def _init_sets(self):
        if self.desc_changed_idx is not None:
            self.changed_idx_set = pr.FrozenBitMap(self.desc_changed_idx)
        else:
            self.changed_idx_set = pr.FrozenBitMap()

        self.ignore_set = self.dataset.seen_indices.union(self.changed_idx_set)

    @staticmethod
    def from_dataset( dataset : Dataset, weight_matrix : sp.csr_array, gamma : float):
        assert weight_matrix.format == 'csr'
        assert len(dataset.idx2label) == 0, 'not implemented other case'
        ## need to initialize numerator and denominator
        sz = weight_matrix.shape[0]
        numerators=np.zeros(sz)
        denominators=np.zeros(sz)
        initial_gamma=initial_gamma_array(gamma, numerators.shape)

        init_scores = (numerators + initial_gamma) / (denominators + 1)
        init_sort = np.argsort(-init_scores)

        
        return LKNNModel(dataset, gamma=initial_gamma, matrix=weight_matrix, numerators=np.zeros(sz), denominators=np.zeros(sz),  score=init_scores, desc_idx=init_sort, 
        desc_score=init_scores[init_sort], desc_changed_idx=None, desc_changed_score=None)

    def _compute_updated_arrays(self, ids, numerator_delta : int, denominator_delta : int, ret_num_denom=False):
        #change_scores = (change_numerators + self.gamma)/(change_denominators + 1)
        numerators = self.numerators[ids]
        denominators = self.denominators[ids]
        gamma = self.gamma[ids]

        change_scores = numexpr.evaluate('(numerators + numerator_delta + gamma)/(denominators + denominator_delta + 1)')
        desc_order = np.argsort(-change_scores)
        desc_changed_idxs = ids[desc_order]
        desc_changed_scores = change_scores[desc_order]

        if ret_num_denom:
            change_numerators = self.numerators[ids] + numerator_delta 
            change_denominators = self.denominators[ids] + denominator_delta
            return desc_changed_idxs, desc_changed_scores, change_scores,  change_numerators, change_denominators
        else:
            return desc_changed_idxs, desc_changed_scores, change_scores, None, None


    def _condition_shared(self, idx, y, ret_num_denom=False):
        start, end = self.matrix.indptr[idx:idx+2]
        neighbors = self.matrix.indices[start:end]

        curr_label = self.dataset.idx2label.get(idx, None)
        if curr_label is None:
            numerator_delta = y
            denominator_delta = 1
        elif curr_label != y:
            assert False, 'no benchmark scenario should reach this'
            numerator_delta = (y - curr_label)
            denominator_delta = 0
        else:
            assert False, 'no benchmark scenario should reach here'
            numerator_delta = 0
            denominator_delta = 0

        desc_changed_idx, desc_changed_score, score_change, num_change, denom_change = self._compute_updated_arrays(neighbors,
                     numerator_delta, denominator_delta, ret_num_denom=ret_num_denom)
        new_dataset = self.dataset.with_label(idx, y)
        return new_dataset, neighbors, desc_changed_idx, desc_changed_score, score_change, num_change, denom_change

    def condition(self, idx, y) -> 'LKNNModel':
        ''' returns new model
        '''
        #row  = self.matrix[[idx],:] # may include itself, but will ignore these
        #_, neighbors = row.nonzero()
        assert self.desc_changed_idx is None
        assert self.matrix.format == 'csr', 'use arrays directly to make access fast'
        
        new_dataset, neighbors, desc_changed_idx, desc_changed_score, score_change, _, _ = self._condition_shared(idx, y, ret_num_denom=False)
        return LKNNModel(new_dataset, gamma=self.gamma, matrix=self.matrix, numerators=self.numerators, denominators=self.denominators, 
                            score=self.score,
                            desc_idx=self.desc_idx, 
                            desc_score=self.desc_score, 
                            desc_changed_idx=desc_changed_idx, desc_changed_score=desc_changed_score)


    def condition_(self, idx, y):
        assert self.desc_changed_idx is None
        new_dataset, neighbors, desc_changed_idx, desc_changed_score, score_change,num_change, denom_change = self._condition_shared(idx, y, ret_num_denom=True)
        self.dataset = new_dataset
        print(f'{idx=} {neighbors=}')
        self.numerators[neighbors] = num_change
        self.denominators[neighbors] = denom_change
        self.score[neighbors] = score_change

        ## slow. sort everything
        self.desc_idx = np.argsort(-self.score)
        self.desc_score = self.score[self.desc_idx]

        self._init_sets()

    def predict_proba(self, idxs : np.ndarray ) -> np.ndarray:
        basic_score = self.score[idxs]
        assert self.desc_changed_idx is None, 'is this ever called after first round'
        return basic_score

    def _iter_desc_scores(self):
        for (idx, score) in zip(self.desc_idx, self.desc_score):
            assert int(idx) == idx
            if idx not in self.ignore_set:
                yield (idx, score)
            else:
                pass

    def _iter_changed_scores(self):
        for (idx, score) in zip(self.desc_changed_idx, self.desc_changed_score):
            assert int(idx) == idx
            if idx not in self.dataset.seen_indices:
                yield (idx, score)

    def iter_desc(self):
        """ merges both streams in descending order, excluding any already seen elements
        """
        if self.desc_changed_idx is not None:

            iter1 = self._iter_desc_scores()
            iter2 = self._iter_changed_scores()

            idx1, score1 = next(iter1, (-1, -math.inf)) # should be rare but in principle it is possible
            idx2, score2 = next(iter2, (-1, -math.inf)) # could be empty if everything has been seen
            while (idx1 > -1) or (idx2 > -1):
                if score1 >= score2:
                    yield (idx1, score1)
                    idx1, score1 = next(iter1, (-1, -math.inf))
                else:
                    yield (idx2, score2)
                    idx2, score2 = next(iter2, (-1, -math.inf))

            assert (idx1 == -1) and (idx2 == -1), 'how did we get here'
        else:
            iter1 =  self._iter_desc_scores()
            for x in iter1:
                yield x

    def top_k_remaining(self, top_k : int) -> Tuple[np.ndarray, np.ndarray]:
        ## cheap form of finding the top k highest scoring without materializing all scores
        vals = []
        idxs = []
        for i, (idx, val) in enumerate(self.iter_desc()):
            if i >= top_k:
                break

            idxs.append(idx)
            vals.append(val)
        return np.array(idxs), np.array(vals)

    def probability_bound(self, n) -> np.ndarray:
        idxs = self.dataset.remaining_indices()
        prob_bounds = (self.gamma + n + self.numerators[idxs])/(1 + n + self.denominators[idxs])
        return np.max(prob_bounds)