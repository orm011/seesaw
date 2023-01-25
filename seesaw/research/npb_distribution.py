import torch
import torch.distributions


class NPBDistribution:
    """  params r, probs=[p1,p2,...]
    X =  number of biased coin tosses (bias is probability for each coin, in their specific order) that we tossed in the specified order until we find r heads.
    X ~ Negative Poisson Binomial (r, probs)
    """
    
    def __init__(self, r, probs):
        self.r = r
        
        ## pad probs to make this always solvable
        self.probs = probs
        self.dist = torch.distributions.Binomial(probs=probs)
    
    def pmf(self, k):
        """
        formula 1: recursive call to pb
        """
        return self.probs[k-1] * PBDistribution(self.probs[:k-1]).pmf(k-1)
        
    def sample(self, sample_shape=torch.Size([])):
        heads = self.dist.sample(sample_shape)
        csum = heads.cumsum(dim=-1)
        first = (csum < self.r).sum(dim=-1) + 1
        return first
        
    def _expectation_accu(self):
        csum = self.probs.cumsum(dim=-1) 
        first_crossing = (csum < self.r).sum(dim=-1)
        ## for r = 1. if p[0] == 1, first_crossing is 0.
        ## for r = 1. if p[0] == .3, first crossing is 3
        # 1/.3 = 3.33
        ## will return 4, overestimate.
        return first_crossing + 1
        
    def _expectation_accu_prime(self):
        mth = self._expectation_accu()
        index = mth - 1 # zero based 
        excess = self.probs[:index + 1].sum() - self.r
        excess2 = self.probs[:index].sum() - self.r
        assert excess >= 0
        assert excess2 < 0
        adjustment = excess / self.probs[index] # fraction of excess
        return mth - adjustment
        
    def expectation(self, method='accu_prime'):
        if method == 'accu_prime':
            return self._expectation_accu_prime()
        elif method == 'accu':
            return self._expectation_accu()
        else:
            assert False

        ### P(k > 0) + P(k > 1) + P(k > r) + P(k > r+1)
    def variance(self):
        pass
        
class PBDistribution:
    """ params probs=[p1,p2,..., p_n]
    X = number of Heads we expect from tossing all n biased coins.
    X ~ Poisson Binomial (probs)
    """
    
    def __init__(self, probs):
        self.probs = probs
        self.dist =  torch.distributions.Binomial(probs=probs)
        
    def pmf(self, k):
        ### prob k seems tricky...
        pass
        
        
    def sample(self, sample_shape=torch.Size([])):
        samps = self.dist.sample(sample_shape)
        return samps.sum(dim=-1)
    
    def expectation(self):
        return self.probs.sum()
    
    def variance(self):
        pass


def test_npb_expectation():
    r = 5
    n_points = 200
    sample_p = torch.distributions.Beta(20, 200).sample((n_points,))
    idxs = torch.argsort(-sample_p)
    sample_p = sample_p[idxs]
    assert sample_p.sum() > 2*r, 'make sure ps can add up to desired target'
    
    cost_dist = NPBDistribution(r, sample_p)
    n_samp = 10000
    n_reps = 40
    
    measured = []
    for reps in range(n_reps):
        mu = cost_dist.sample((n_samp,)).float().mean()
        measured.append(mu)
        
    mm = torch.tensor(measured)
    estimate1 = cost_dist.expectation(method='accu_prime')
    estimate0 = mm.mean()
    deviation =  (estimate1 - estimate0).abs()
    assert deviation < max(3*mm.std().item(), 1)