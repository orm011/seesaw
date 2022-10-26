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

def normalize_scores(scores, epsilon = .1):
    assert epsilon < .5
    gap = scores.max() - scores.min()
    if gap == 0: # center at .5 is all scores the same
        return scores - scores + .5
    
    x = (scores - scores.min()) / (scores.max() - scores.min())
    x = x*(1-2*epsilon) + epsilon # shift to be between (epislon, 1-epsilon)
    return x

class BaseLabelPropagationRanker:
    def __init__(self, *, knng : KNNGraph, normalize_scores, calib_a, calib_b, prior_weight, edist, num_iters, **other):
        self.knng = knng
        self.nvecs = knng.nvecs
        nvecs = self.nvecs
        self.normalize_scores = normalize_scores
        self.calib_a = calib_a
        self.calib_b = calib_b
        self.prior_weight = prior_weight
        self.edist = edist
        self.num_iters = num_iters
        self.epsilon = .3

        self.is_labeled = np.zeros(nvecs)
        self.labels = np.zeros(nvecs)
        self._scores = None

        self.all_indices = pr.FrozenBitMap(range(nvecs))

    def set_base_scores(self, init_scores):
        assert self.nvecs == init_scores.shape[0]
        ## 1. normalize scores to fit between 0.1 and 0.9

        if self.normalize_scores:
            scores = normalize_scores(init_scores, epsilon=self.epsilon)
        else:
            scores = init_scores

        self.prior_scores = scores # sigmoid(self.calib_a*(scores + self.calib_b))
        self._scores = self.prior_scores.copy()
        self._scores = self._propagate(num_iters=self.num_iters)

    def _propagate(self, num_iters):
        raise NotImplementedError('implement me')

    def update(self, idxs, labels):
        for idx, label in zip(idxs, labels):
            idx = int(idx)
            label = float(label)
            assert np.isclose(label,0) or np.isclose(label,1)
            self.labels[idx] = label  # make 0 or 1
            self.is_labeled[idx] = 1
                
        pscores = self._propagate(self.num_iters)
        self._scores = pscores

    def current_scores(self):
        return self._scores

    def top_k(self, k, unlabeled_only=True):
        if unlabeled_only:
            subset = np.where(self.is_labeled < 1)[0]
        else: 
            subset = np.array(self.all_indices)
            
        raw_scores = self.current_scores()        
        topk_positions = np.argsort(-raw_scores[subset])[:k]
        topk_indices = subset[topk_positions]
        return topk_indices, raw_scores[topk_indices]


class LabelPropagationRanker(BaseLabelPropagationRanker):
    def __init__(self, knng : KNNGraph, **super_args):
        super().__init__(knng=knng, **super_args)
        self.adj_mat, self.norm_w = prepare(self.knng, prior_weight=self.prior_weight, edist=self.edist)

    def _propagate(self, num_iters):
        labeled_prior = self.prior_scores * (1-self.is_labeled) + self.labels * self.is_labeled
        scores = self._scores # use previous scores as starting point
        for _ in range(num_iters): 
            prev_score = scores
            subtot = (self.adj_mat @ prev_score) + self.prior_weight*labeled_prior
            scores = subtot*self.norm_w

            ## override scores with labels 
            scores = scores * (1 - self.is_labeled) + self.labels * self.is_labeled
            # norm = np.linalg.norm(prev_score - scores)
            # print(f'norm delta : {norm}')

        return scores

def _prepare_propagation_matrices(weight_matrix, reg_lambda):
    # fprime = (D + reg_lambda * I)^-1 (W @ f + reg_lambda * p)
    # let inv_D = (D + reg_lambda * I)^-1
    # fprime = (inv_D @ W) @ f + reg_lambda * (inv_D @ p)
    # precompute (inv_D @ W) as   normalized weights 
    # and 
    # reg_lambda * (inv_D @ p) as normalized_prior
    assert reg_lambda >= 0
    n = weight_matrix.shape[0]

    weights = weight_matrix.sum(axis=0) + reg_lambda
    reg_inv_weights = 1./weights
    normalizer_matrix = sp.coo_array( (reg_inv_weights, (np.arange(n), np.arange(n))), shape=(n,n) )

    normalized_weights = normalizer_matrix.tocsr() @ weight_matrix.tocsc()
    prior_normalizer = reg_lambda * normalizer_matrix

    # assert np.isclose(normalized_weights.sum(axis=1), normalized_weights.sum(axis=0)).all()
    # assert np.isclose(normalized#_weights.sum(axis=1) + reg_lambda, 1).all()

    return normalized_weights.tocsr(), prior_normalizer.tocsr()


class LabelPropagation:
    def __init__(self, weight_matrix, *, reg_lambda : float, max_iter : int, epsilon=1e-4, verbose=0):
        assert reg_lambda >= 0

        self.weight_matrix = weight_matrix
        n = self.weight_matrix.shape[0]
        self.n = n
        self.epsilon = epsilon
                
        self.verbose = verbose
        self.reg_lambda = reg_lambda
        
        self.max_iter = max_iter
        self.normalized_weights, self.prior_normalizer = _prepare_propagation_matrices(weight_matrix, reg_lambda)
        self.normalized_prior = None

    def _loss(self, label_ids, label_values):
        pass
        
    def _step(self, old_fvalues, label_ids, label_values):
        new_fvalues = self.normalized_weights @ old_fvalues + self.normalized_prior
        assert old_fvalues.min() <= new_fvalues.min(), 'propgation should smoothen scores toward the middle. check weights'
        assert new_fvalues.max() <= old_fvalues.max(), 'averaged scores should lie within previous scores. check weights '

        new_fvalues[label_ids] = label_values
        return new_fvalues

    def fit_transform(self, *, label_ids, label_values, reg_values, start_value=None):
        assert reg_values.shape[0] == self.n
        self.normalized_prior = self.prior_normalizer @ reg_values

        if start_value is not None:
            old_fvalues = start_value.copy()
        else:
            old_fvalues = reg_values.copy()
            
        old_fvalues[label_ids] = label_values
        
        converged = False

        i = 0
        for i in range(1, self.max_iter+1):
            new_fvalues = self._step(old_fvalues, label_ids, label_values)

            if np.max((new_fvalues - old_fvalues)**2) < self.epsilon:
                converged = True
                if self.verbose > 0:
                    print(f'converged after {i} iterations')
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
        self.normalized_weights_intra, self.prior_normalizer_intra = _prepare_propagation_matrices(weight_matrix_intra, self.reg_lambda)

    def fit_transform(self, *, reg_values, **kwargs):
        self.normalized_prior_intra = self.prior_normalizer_intra @ reg_values
        return super().fit_transform(reg_values=reg_values, **kwargs)

    def _step(self, old_fvalues, label_ids, label_values):
        new_fvalues = self.normalized_weights @ old_fvalues + self.normalized_prior
        new_fvalues[label_ids] = label_values

        new_fvalues = self.normalized_weights_intra @ new_fvalues + self.normalized_prior_intra
        new_fvalues[label_ids] = label_values
        return new_fvalues


class LabelPropagationRanker2(BaseLabelPropagationRanker):
    lp : LabelPropagation

    def __init__(self, *, knng_intra : KNNGraph = None, knng : KNNGraph, self_edges : bool, normalized_weights : bool, **other):
        super().__init__(knng=knng, **other)
        self.knng_intra = knng_intra

        kfun = lambda x : kernel(x, self.edist)
        self.weight_matrix = get_weight_matrix(knng, kfun, self_edges=self_edges, normalized=normalized_weights)
        common_params = dict(reg_lambda = self.prior_weight, weight_matrix=self.weight_matrix, max_iter=self.num_iters)
        if knng_intra is None:
            self.lp = LabelPropagation(**common_params)
        else:
            self.weight_matrix_intra = get_weight_matrix(knng_intra, kfun, self_edges=self_edges, normalized=normalized_weights)
            self.lp = LabelPropagationComposite(weight_matrix_intra = self.weight_matrix_intra, **common_params)
    
    def _propagate(self, num_iters):
        ids = np.nonzero(self.is_labeled.reshape(-1))
        labels = self.labels.reshape(-1)[ids]
        scores = self.lp.fit_transform(label_ids=ids, label_values=labels, reg_values= self.prior_scores, start_value=self._scores)
        return scores