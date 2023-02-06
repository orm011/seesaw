
from .common import IncrementalModel, Result
import numpy as np
import torch
import math

def _expected_utility_approx(t: int, model : IncrementalModel):
    idxs, scores = model.top_k_remaining(top_k=t)
    next_idx = idxs[0]
    expected_u = scores.sum()
    return Result(value=expected_u, index=next_idx)


def opt_expected_utility(remaining_lookahead: int, t : int, model : IncrementalModel):
    '''l: lookahead exact horizon
       t: lookahead total horizon
       
       returns the expected utlity at horizon t for this batch, assuming an exact
       look-ahead of k <= t
    '''
    if remaining_lookahead == 0:
        return _expected_utility_approx(t, model)


    idxs = model.dataset.remaining_indices()
    p1 = model.predict_proba(idxs)
    probs = np.concatenate([1-p1, p1], axis=-1)
    values = np.zeros_like(probs)
    for i,idx in enumerate(idxs):
        cond0 = model.with_label(idx, 0)
        cond1 = model.with_label(idx, 1)
        util0 = opt_expected_utility(remaining_lookahead - 1, t, cond0)
        util1 = opt_expected_utility(remaining_lookahead - 1, t, cond1)
        values[i,0] = util0.value
        values[i,1] = util1.value

    expected_utils = (probs * values).sum(axis=-1)
    assert expected_utils.shape[0] == values.shape[0]
    pos = np.argmax(expected_utils)
    return Result(value=expected_utils[pos], index=idxs[pos])
