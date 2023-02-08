from  seesaw.research.npb_distribution import NPBDistribution
import math
import numpy as np
import torch

from .common import ProbabilityModel, Result


def expected_cost(idx, *, r : int,  t : int,  model : ProbabilityModel) -> float:
    p = model.predict_proba(np.array([idx])).item()
    # case y = 1
    res1 = min_expected_cost_approx(r-1,  t=t-1, model = model.condition(idx,1))

    # case y = 0
    res0 = min_expected_cost_approx(r,  t=t-1, model = model.condition(idx,0))
    
    return p*res1.value + (1-p)*res0.value

def min_expected_cost_approx(r : int, *,  top_k : int = None, t : int, model : ProbabilityModel) -> Result:
    if t == 1:
        indices = model.dataset.remaining_indices()
        probs = model.predict_proba(indices)
        probs = torch.from_numpy(probs)
        desc_order = torch.argsort(-probs)
        desc_probs = probs[desc_order]        
        cost =  NPBDistribution(r, desc_probs).expectation(method='accu_prime')
        index = indices[desc_order[0]]
        return Result(value=cost, index=index)

    min_idx = None
    min_cost = math.inf

    idxs = model.dataset.remaining_indices()
    curr_pred = model.predict_proba(idxs)
    desc_order = np.argsort(-curr_pred)
    top_k_idxs = idxs[desc_order[:top_k]]

    for idx in top_k_idxs:
        c = expected_cost(idx, r=r, t=t, model=model)
        if c < min_cost:
            min_idx = idx
            min_cost = c
            
    return Result(value=min_cost, index=min_idx)