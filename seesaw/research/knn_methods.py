import numpy as np
import pyroaring as pr
from seesaw.knn_graph import *

from scipy.special import expit as sigmoid

class SimpleKNNRanker:
    def __init__(self, knng, init_scores=None):
        self.knng : KNNGraph = knng

        if init_scores is None:
            self.init_numerators = np.ones(self.knng.nvecs)*.1 # base if nothing is given
        else:
            self.set_base_scores(init_scores)

        self.pscount = 1.
        
        self.numerators = np.zeros_like(self.init_numerators)
        self.denominators = np.zeros_like(self.init_numerators)

        self.labels = np.zeros_like(self.init_numerators)
        self.is_labeled = np.zeros_like(self.init_numerators)
        
        self.all_indices = pr.FrozenBitMap(range(self.knng.nvecs))
        
    def current_scores(self):
        num = self.pscount*self.init_numerators + self.numerators
        denom = self.pscount + self.denominators
        estimates = num/denom
        return self.labels*self.is_labeled + estimates*(1-self.is_labeled)
        
    def set_base_scores(self, scores):
        assert self.knng.nvecs == scores.shape[0]
        self.init_numerators = sigmoid(2*scores)

    def update(self, idxs, labels):
        for idx, label in zip(idxs, labels):
            idx = int(idx)
            label = float(label)
            
            assert np.isclose(label,0) or np.isclose(label,1)
            
            if self.is_labeled[idx] > 0: # if new label for old 
                old_label = self.labels[idx]
                delta_denom = 0
                delta_num = label - old_label # erase old label and add new label
            else:
                delta_num = label
                delta_denom = 1
            
            self.labels[idx] = label
            self.is_labeled[idx] = 1
                    
            ## update scores for all v such that idx \in knn(v)
            rev_neighbors = self.knng.rev_lookup(idx).src_vertex.values
            # rev_weights = 
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


from scipy.special import expit as sigmoid
import numpy as np


def prepare(knng : KNNGraph, *, edist, prior_weight):
    knndf = knng.knn_df 
    symknn = knndf.assign(weight = kernel(knndf.distance, edist=edist))
    n = knng.nvecs

    wmatrix = sp.coo_matrix( (symknn.weight.values, (symknn.src_vertex.values, symknn.dst_vertex.values)), shape=(n, n))
    diagw = sp.coo_matrix((np.ones(n)*prior_weight, (np.arange(n), np.arange(n))))
    wmatrix_tot = wmatrix + diagw
    norm_w = 1./np.array(wmatrix_tot.sum(axis=1)).reshape(-1)
    adj_matrix = wmatrix.tocsr()
    return adj_matrix, norm_w

from sklearn.metrics import average_precision_score

def normalize_scores(scores, epsilon):
    assert epsilon < .5
    gap = scores.max() - scores.min()
    if gap == 0: # center at .5 is all scores the same
        return scores - scores + .5
    
    x = (scores - scores.min()) / (scores.max() - scores.min())
    x = x*(1-2*epsilon) + epsilon # shift to be between (epislon, 1-epsilon)
    return x


class BaseLabelPropagationRanker:
    def __init__(self, *, knng : KNNGraph, normalize_scores, sigmoid_before_propagate, calib_a, calib_b, prior_weight, edist, normalize_epsilon = None, **other):
        self.knng = knng
        self.nvecs = knng.nvecs
        nvecs = self.nvecs
        self.normalize_scores = normalize_scores

        if self.normalize_scores:
            assert normalize_epsilon is not None
            self.epsilon = normalize_epsilon

        self.calib_a = calib_a
        self.calib_b = calib_b
        self.prior_weight = prior_weight
        self.edist = edist
        self.sigmoid_before_propagate = sigmoid_before_propagate

        self.is_labeled = np.zeros(nvecs)
        self.labels = np.zeros(nvecs)

        self.prior_scores = None
        self._current_scores = None

        self.all_indices = pr.FrozenBitMap(range(nvecs))

    def set_base_scores(self, init_scores):
        assert self.nvecs == init_scores.shape[0]
        ## 1. normalize scores to fit between 0.1 and 0.9

        if self.normalize_scores:
            init_scores = normalize_scores(init_scores, epsilon=self.epsilon)

        if self.sigmoid_before_propagate:# affects the size of the scores wrt. 0, 1 labels from user.
            ## also affects the regularization target things are pushed back to.
            self.prior_scores = sigmoid(self.calib_a*(init_scores + self.calib_b))
        else:
            self.prior_scores = init_scores 

        self._current_scores = self._propagate(self.prior_scores)

    def _propagate(self, scores):
        raise NotImplementedError('implement me')

    def update(self, idxs, labels):
        for idx, label in zip(idxs, labels):
            idx = int(idx)
            label = float(label)
            assert np.isclose(label,0) or np.isclose(label,1)
            self.labels[idx] = label  # make 0 or 1
            self.is_labeled[idx] = 1
                
        pscores = self._propagate(self.prior_scores)
        self._current_scores = pscores

    def current_scores(self):
        return self._current_scores

    def top_k(self, k, unlabeled_only=True):
        if unlabeled_only:
            subset = np.where(self.is_labeled < 1)[0]
        else: 
            subset = np.array(self.all_indices)
            
        raw_scores = self.current_scores()        
        topk_positions = np.argsort(-raw_scores[subset])[:k]
        topk_indices = subset[topk_positions]
        return topk_indices, raw_scores[topk_indices]


class LabelPropagation:
    def __init__(self, weight_matrix, *, reg_lambda : float, max_iter : int, epsilon=1e-5, verbose=0):
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

        lower_bounds = np.minimum(old_fvalues.min(), self.reg_values)
        upper_bounds = np.maximum(old_fvalues.max(), self.reg_values)

        assert (lower_bounds <= new_fvalues).all(), 'propgation should smoothen scores toward the middle. check weights'
        assert (new_fvalues <= upper_bounds).all(), 'averaged scores should lie within previous scores. check weights '

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

    
class LabelPropagationComposite(LabelPropagation):
    def __init__(self, *, weight_matrix_intra, **other_kwargs):
        super().__init__(**other_kwargs)
        self.weight_matrix_intra = weight_matrix_intra

    def fit_transform(self, *, reg_values, **kwargs):
        self.normalized_prior_intra = reg_values
        return super().fit_transform(reg_values=reg_values, **kwargs)

    def _step(self, old_fvalues, label_ids, label_values):
        new_fvalues = self.normalized_weights @ old_fvalues + self.normalized_prior
        new_fvalues[label_ids] = label_values

        new_fvalues = self.normalized_weights_intra @ new_fvalues + self.normalized_prior_intra
        new_fvalues[label_ids] = label_values
        return new_fvalues


class LabelPropagationRanker2(BaseLabelPropagationRanker):
    lp : LabelPropagation

    def __init__(self, *, knng_intra : KNNGraph = None, knng : KNNGraph, self_edges : bool, normalized_weights : bool, verbose : int = 0, **other):
        super().__init__(knng=knng, **other)
        self.knng_intra = knng_intra

        kfun = rbf_kernel(self.edist)
        print('getting weight matrix')
        self.weight_matrix = get_weight_matrix(knng.knn_df, kfun=kfun, self_edges=self_edges, normalized=normalized_weights)
        print('got weight matrix')
        common_params = dict(reg_lambda = self.prior_weight, weight_matrix=self.weight_matrix, max_iter=300, verbose=verbose)
        if knng_intra is None:
            self.lp = LabelPropagation(**common_params)
        else:
            self.weight_matrix_intra = get_weight_matrix(knng_intra, kfun=kfun, self_edges=self_edges, normalized=normalized_weights)
            self.lp = LabelPropagationComposite(weight_matrix_intra = self.weight_matrix_intra, **common_params)
    
    def _propagate(self,  scores):
        ids = np.nonzero(self.is_labeled.reshape(-1))
        labels = self.labels.reshape(-1)[ids]
        scores = self.lp.fit_transform(label_ids=ids, label_values=labels, reg_values= self.prior_scores, start_value=scores)
        return scores