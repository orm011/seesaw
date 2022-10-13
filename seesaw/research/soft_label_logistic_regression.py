import cvxpy as cp
from scipy.special import expit

def bce_loss(p, q): # p is 'true' prob
    # - p log (q) - (1-p) log(1-q)
    return cp.multiply(p,cp.log(q)) + cp.multiply(1-p, cp.log(1-q))

def bce_loss_with_logits(p, logits):
    # NB. 
    ## sigmoid(score) = 1/(1 + exp(-score)). 
    ### as score -> infty , this goes to 1.
    # if q = sigmoid(score)
    # then 1 - q = sigmoid(-score)
    # hence,
    # log(q) = log(sigmoid(score)) = -log(1+exp(-score))
    # log(1-q) = -log(1+exp(score))
    # p log(1+exp(-score)) + (1-p) log(1+exp(score))
    return cp.multiply(p, cp.logistic(-logits)) + cp.multiply(1-p, cp.logistic(logits))

class LinearModelCE:
    def __init__(self, *, solver_flags, C=1., class_weights='balanced', regularizer_vector=None):
        self.theta = None 
        self.bias = None
        self.regularizer_vector = regularizer_vector
        self.class_weights = class_weights
        self.problem = None
        self.C = C
        self.solver_flags = solver_flags
        
    def _loss(self, scores, ps, weights=None):
        losses = bce_loss_with_logits(ps, scores)
        if weights is None:
            weights = np.ones_like(losses)

        weights = weights / weights.sum()

        return losses @ weights

    def _regularizer(self):
        if self.regularizer_vector is None:
            return cp.norm(self.theta)
        else:
            return cp.norm(self.theta - self.regularizer_vector)
        
    def _init_problem(self, X, ps):
        self.theta = cp.Variable((X.shape[1],))
        self.bias = cp.Variable((1,))
        num_pos = (ps == 1).sum()
        wpos = (ps.shape[0] - num_pos)/ num_pos

        weights = np.ones_like(ps)
        weights[ps == 1] = wpos
        
        scores = self._scores(X)  
        loss = self._loss(scores, ps, weights)
        reg = self._regularizer()

        rloss =  self.C*loss + reg
        self.problem = cp.Problem(cp.Minimize(rloss))

        
    def _scores(self, X, eval_mode=False):
        if eval_mode:
            theta = self.theta.value
            bias = self.bias.value
        else:
            theta = self.theta
            bias = self.bias
            
        return X @ theta + bias
    
    def fit(self, X,ps):
        ps = ps.reshape(-1)
        self._init_problem(X,ps)
        self.problem.solve(**self.solver_flags)
        
    def predict_proba(self, X):
        scores = self._scores(X, eval_mode=True)
        return expit(scores)

from .knn_methods import LabelPropagationRanker
from sklearn.decomposition import PCA 
import numpy as np
import pyroaring as pr

def makeXy(idx, lr, sample_size, pseudoLabel=True):

    Xlab = idx.vectors[(lr.is_labeled > 0) ]
    ylab = lr.labels[(lr.is_labeled > 0) ]
    
    rsize = sample_size - Xlab.shape[0]

    scores = lr.current_scores()
    rsample = np.random.permutation(idx.vectors.shape[0])[:rsize]

    if pseudoLabel:

        Xsamp = idx.vectors[rsample]
        ysamp = scores[rsample]
        
        X = np.concatenate((Xlab, Xsamp))
        y = np.concatenate((ylab, ysamp))
        
        # if quantile_transform:
        #     ls = QuantileTransformer()
        #     ls.fit(scores.reshape(-1,1))
        #     y = ls.transform(y.reshape(-1,1)).reshape(-1)
    else:
        X = Xlab
        y = ylab
        
    return X,y


_default_linear_args=dict(C=10., solver_flags=dict(solver=cp.MOSEK, verbose=False))
_default_pca_args=dict(n_components=128, whiten=True)
_default_prop_args=dict(calib_a=10., calib_b=-5, prior_weight=1., edist=.1, num_iters=5)

class LinearScorer:
    def __init__(self, idx, knng_sym, *, init_scores=None, sample_size, 
                    label_prop_kwargs=_default_prop_args, 
                    pca_kwargs=_default_pca_args, linear_kwargs=_default_linear_args, **other):
        self.idx = idx
        self.knng_sym = knng_sym
        self.sample_size = sample_size
        self.label_prop_kwargs = label_prop_kwargs
        self.pca_kwargs= pca_kwargs
        self.linear_kwargs = linear_kwargs

        if init_scores is not None:
            self._finish_init(init_scores)

    def _finish_init(self, init_scores):
        self.lr = LabelPropagationRanker(knng=self.knng_sym, init_scores=init_scores, **self.label_prop_kwargs)
        pca = PCA(**self.pca_kwargs)
        samp = np.random.permutation(self.idx.vectors.shape[0])[:10000]
        Xsamp = self.idx.vectors[samp]
        pca.fit(Xsamp)
        self.all_indices = pr.BitMap(range(self.idx.vectors.shape[0]))
        self.seen = pr.BitMap()
        self.pca = pca
        self.Xtransformed = pca.transform(self.idx.vectors)
        self.lm = LinearModelCE(**self.linear_kwargs)
        self._update_lm()

    def set_base_scores(self, init_scores):
        self._finish_init(init_scores)

    def _update_lm(self):
        X, y = makeXy(self.idx, self.lr, sample_size=self.sample_size, pseudoLabel=True)
        Xtrans = self.pca.transform(X)
        self.lm.fit(Xtrans, y)
        
    def update(self, idxs, labels):
        self.lr.update(idxs, labels)
        self._update_lm()
        for idx in idxs:
            self.seen.add(idx)

    def current_scores(self):
        return self.lm.predict_proba(self.Xtransformed).reshape(-1)
    
    def top_k(self, k, unlabeled_only=True):
        if unlabeled_only:
            subset = np.array(self.all_indices - self.seen)
        else: 
            subset = np.array(self.all_indices)
            
        raw_scores = self.current_scores()
        topk_positions = np.argsort(-raw_scores[subset])[:k]
        topk_indices = subset[topk_positions]
        return topk_indices, raw_scores[topk_indices]


