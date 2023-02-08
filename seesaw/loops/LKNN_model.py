    
from ..research.active_search.common import ProbabilityModel, Dataset
import numpy as np
import scipy.sparse as sp

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
        print(f'{matrix.shape=}')

        ## set probs to estimates, then replace estimates with labels
        self._probs = (gamma + numerators) / (1 + denominators)

        if len(dataset.seen_indices) > 0:
            idxs, labels = dataset.get_labels()
            self._probs[idxs] = labels


    @staticmethod
    def from_dataset( dataset : Dataset, weight_matrix : sp.csr_array, gamma : float):
        assert weight_matrix.format == 'csr'
        assert len(dataset.idx2label) == 0, 'not implemented other case'
        ## need to initialize numerator and denominator
        sz = weight_matrix.shape[0]
        return LKNNModel(dataset, gamma=gamma, matrix=weight_matrix, numerators=np.zeros(sz), denominators=np.zeros(sz))


    def condition(self, idx, y) -> 'LKNNModel':
        ''' returns new model
        '''

        numerators = self.numerators.copy()
        denominators = self.denominators.copy()


        row  = self.matrix.getrow(idx) # may include itself, but will ignore these
        _, neighbors = row.nonzero()
        #neighbors = neighbors.reshape(-1)
        print(neighbors)

        curr_label = self.dataset.idx2label.get(idx, None)
        if curr_label is None:
            numerators[neighbors] += y
            denominators[neighbors] += 1
        elif curr_label != y:
            numerators[neighbors] += (y - curr_label)
        else: # do nothing.
            pass

        new_dataset = self.dataset.with_label(idx, y)
        return LKNNModel(new_dataset, gamma=self.gamma, matrix=self.matrix, numerators=numerators, denominators=denominators)

    def predict_proba(self, idxs : np.ndarray ) -> np.ndarray:
        return self._probs[idxs]

    def probability_bound(self, n) -> np.ndarray:
        idxs = self.dataset.remaining_indices()
        prob_bounds = (self.gamma + n + self.numerators[idxs])/(1 + n + self.denominators[idxs])
        return np.max(prob_bounds)