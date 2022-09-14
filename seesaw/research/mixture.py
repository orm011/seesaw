# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
### notes: kmeans ++ initialization is available in sklearn, 
#### as is a dirchlet process prior with no fixed number of clusters.
## for fitting, they do not use EM.
### we can reasonably try implmenting a MAP version with labels using EM.
### the algorithm is fully unsupervised, though.
### so they cannot integrate label information from some elements

from einops import rearrange, einsum
import torch

import numpy as np
import math
import pandas as pd
import torch
import torch.distributions as dist

from torch.nn import functional as F
from ray.data.extensions import TensorArray

def gen_data(n_classes = 3, n_dim = 2, n_samples=1000, normalize=False):
    meta_dist = dist.MultivariateNormal(loc=torch.zeros(n_dim), covariance_matrix=4*torch.eye(n_dim))
    mus = meta_dist.sample(sample_shape=(n_classes,))
    covars = torch.stack([torch.eye(n_dim) for i in range(n_classes)])

    class_conditional_dist = [dist.MultivariateNormal(loc=mus[i], covariance_matrix=covars[i]) for i in range(n_classes)]
    class_probs = dist.Dirichlet(torch.ones(n_classes)).sample()
    class_probs = torch.sort(class_probs)[0] # class 0 is the least popular
    class_dist = dist.Categorical(class_probs)

    cats = class_dist.sample((n_samples,))
    points = torch.stack([class_conditional_dist[c].sample() for c in cats])
    
    if normalize:
        points = F.normalize(points)
        
    # cats = pd.get_dummies(cats).to_numpy()
    df= pd.DataFrame({'X':TensorArray(points.numpy()), 'c':cats})
    return df, class_probs, mus, covars

def _mahalanobis_norm(X, mus, inv_cov):
    X_tmp = rearrange(X, 'item d -> () item d') # align both to subtract
    mu_tmp = rearrange(mus, 'k d -> k () d')
    Xdiff = X_tmp - mu_tmp
    
    Xdiffprime = einsum(Xdiff, inv_cov, 'k item i, k i j -> k item j')
    dist = einsum(Xdiffprime, Xdiff, 'k item j, k item j -> item k')
    return dist


class MixtureModel:
    def __init__(self, n_components, manual_seed=None):
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
        self.n_components = n_components
        ## sets some initial values

    def _init_params(self, Xs):
        d = Xs.shape[1]
        k = self.n_components
        self.log_pz = torch.ones(k).div(k).log().reshape(1,k) # k
        self.mus = torch.randn(k,d) # k by d
        self.cov = torch.stack([torch.eye(d) for i in range(k)]) # k x d x d

    def inv_cov(self):
        return self.cov.inverse()

    def _fit_theta(self, Xs, ps):        
        assert (ps >= 0).all()
        assert torch.isclose(ps.sum(dim=-1), torch.tensor(1.)).all()

        nitems = Xs.shape[0]        
        nitems = torch.tensor(nitems).float()        
        
        ## fit pis
        ps = ps/nitems
        _pzs = einsum(ps, 'item c -> c')
        self.log_pz = _pzs.log()
        
        # now fit mus and covs
        norm_ps = ps/ps.sum(dim=0, keepdims=True)
        assert torch.isclose(norm_ps.sum(dim=0), torch.tensor(1.)).all()
        
        norm_ps = norm_ps.t()
        
        ## weighted sum  of mus and covs
        self.mus = einsum(norm_ps, Xs,  'k item, item d -> k d')
        
        # align the singleton dimensions for broadcast all means from all points
        X0 = rearrange(Xs, 'item d -> () item d') 
        mu0 = rearrange(self.mus, 'k d -> k () d')
        Xdiff = X0 - mu0 # k x item x d
        
        self.cov = einsum(Xdiff, Xdiff, norm_ps, 'k item i, k item j, k item -> k i j')
        
    def fit(self, Xs, max_iters=100):
        self._init_params(Xs)
        lls = []
        ll = self._log_px(Xs).mean()
        lls.append(ll)
        ps = self._log_pz_given_x(Xs).exp()
        for i in range(max_iters):
            self._fit_theta(Xs, ps)
            ll = self._log_px(Xs).mean()
            lls.append(ll)
            ps = self._log_pz_given_x(Xs).exp()
        return lls
    
    def _log_pz_given_x(self, Xs):
        log_px_given_z = self._log_px_given_z(Xs)
        log_px =  self.log_pz + log_px_given_z # whats the right name to use here
        return log_px.log_softmax(dim=-1)
    
    def _log_px(self, Xs): 
        ### log likelihood of the observed data according to the model.
        # p(x ; theta) = \sum_z=1 to k p( x | z ; theta) p(z ; theta) 
        # hence
        # log p(x ; theta) = log sum_z   exp  (log p(x | z ; theta) + log p (z ; theta))
        log_px_given_z = self._log_px_given_z(Xs) 
        # the above is possibly off by a constant wrt. z. this is ok  for pz_given_x,
        ## but could it affect the next computation?
        log_px = torch.logsumexp(self.log_pz + log_px_given_z, dim=-1)
        return log_px

    def _log_normalization_constants(self): 
        # log ( 1/  sqrt (2 * pi * det(\Sigma) ))
        # -.5 * log (2*pi* det(sigma))
        return self.cov.det().mul(2*math.pi).log().mul(-.5)  # off by -.9189 = -.5*.mul(2*math.pi), which is constant, to be consistent with sklearn
 
    def _mahalanobis_norm(self, Xs):
        return _mahalanobis_norm(Xs, self.mus, self.inv_cov())
                
    def _log_px_given_z(self, Xs):
        return self._log_normalization_constants().reshape(1,-1) - .5 *self._mahalanobis_norm(Xs)