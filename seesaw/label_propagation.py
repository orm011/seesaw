import numpy as np
import scipy.sparse as sp

from seesaw.knn_graph import *

class LabelPropagation:
    def __init__(self, weight_matrix : sp.csr_array, *, reg_lambda : float, max_iter : int, epsilon=1e-5, verbose=0):
        assert reg_lambda >= 0

        self.weight_matrix = weight_matrix
        n = self.weight_matrix.shape[0]
        self.n = n
        self.epsilon = epsilon
                
        self.verbose = verbose
        self.reg_lambda = reg_lambda
        
        self.max_iter = max_iter

        csr_format = weight_matrix
        assert csr_format.has_sorted_indices
        self.weight_matrix = csr_format
        
        self.reg_values = None
        self.weight_sum = weight_matrix.sum(0)

    def _loss(self, label_ids, label_values):
        pass
        
    def _step(self, old_fvalues, label_ids, label_values):
        weighted_fvalues = self.weight_matrix @ old_fvalues + (self.reg_lambda * self.reg_values)
        new_fvalues = weighted_fvalues / (self.weight_sum + self.reg_lambda)

        # we have seen bugs where the values seem to diverge due to faulty normalization.
        ## each final score is a weighted average of all the neighbors and its prior. therefore it is always within the two extremes
        low_bound = min(0, self.reg_values.min())
        high_bound = max(1., self.reg_values.max())

        assert (new_fvalues >= low_bound).all(), 'averaged scores should lie at or above 0'
        assert (new_fvalues <= high_bound).all(), 'averaged scores should lie at or below 1'

        new_fvalues[label_ids] = label_values
        return new_fvalues

    def fit_transform(self, *, label_ids, label_values, reg_values = None, start_value=None):
        if reg_values is not None:
            assert reg_values.shape[0] == self.n
            self.reg_values = reg_values
        else:
            assert self.reg_lambda == 0
            self.reg_values = np.zeros(self.weight_matrix.shape[0])

        if start_value is not None:
            old_fvalues = start_value.copy()
        elif reg_values is not None:
            old_fvalues = reg_values.copy()
        else:
            old_fvalues = np.ones(self.weight_matrix.shape[0])*.0
            
        old_fvalues[label_ids] = label_values
        
        converged = False

        i = 0
        for i in range(1, self.max_iter+1):
            new_fvalues = self._step(old_fvalues, label_ids, label_values)

            if np.max((new_fvalues - old_fvalues)**2) < self.epsilon:
                converged = True
                if self.verbose > 0:
                    print(f'prop. converged after {i} iterations')
                break
            else:
                old_fvalues = new_fvalues

        if not converged:
            print(f'warning: did not converge after {i} iterations')
            
        return old_fvalues
