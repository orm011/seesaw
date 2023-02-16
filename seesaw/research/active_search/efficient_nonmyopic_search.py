
from .common import ProbabilityModel, Result
import numpy as np
import pyroaring as pr

def _expected_utility_approx(t: int, model : ProbabilityModel):
    assert t > 0
    idxs, scores = model.top_k_remaining(top_k=t)
    next_idx = idxs[0]
    expected_u = scores.sum()
    return Result(value=expected_u, index=next_idx, pruned_fraction=None)

def _opt_expected_utility_helper(*, i : int,  lookahead_limit : int, t : int, model : ProbabilityModel, pruning_on : bool):
    '''l: lookahead exact horizon
       t: lookahead total horizon
       
       returns the expected utlity at horizon t for this batch, assuming an exact
       look-ahead of k <= t
    '''

    assert i >= 0
    assert i < lookahead_limit
    if (i == lookahead_limit - 1):
        return _expected_utility_approx(t - i, model)

    idxs = model.dataset.remaining_indices()
    p1 = model.predict_proba(idxs).reshape(-1,1)

    probs = np.concatenate([1-p1, p1], axis=-1)
    assert probs.shape[0] == p1.shape[0]
    assert probs.shape[1] == 2

    def _solve_idx(idx, i):
        util0 = _opt_expected_utility_helper(i=i+1, lookahead_limit=lookahead_limit, t=t, model=model.condition(idx, 0), pruning_on=pruning_on)
        util1 = _opt_expected_utility_helper(i=i+1, lookahead_limit=lookahead_limit, t=t, model=model.condition(idx, 1), pruning_on=pruning_on)
        return np.array([util0.value, util1.value])

    if pruning_on:
        p1 = p1.reshape(-1,1)

        pbound = model.probability_bound(1)
        value_bound1 = 1 + (t - i)*pbound
        top_idxs, top_ps = model.top_k_remaining(top_k=(t - i))
        top_idx = top_idxs[0]
        pval = top_ps[0]
        assert top_ps.shape[0] == t - i, f'{top_ps.shape[0]=} {t-i=}'
        value_bound0 =  top_ps.sum()
        upper_bounds = p1 * value_bound1 + (1-p1) * value_bound0

        lower_bound = _solve_idx(top_idx, i) @ np.array([1-pval, pval])

        pruned = upper_bounds < lower_bound
        pruned_fraction = pruned.sum()/pruned.shape[0]
        kept = pruned.shape[0] - pruned.sum()
        print(f'{pruned_fraction=:.03f} {kept=}')

        pruned = pruned.squeeze()
        positions = np.where(pruned)[0]

        pruned_set = pr.BitMap()
        for pos in positions:
            pruned_set.add(idxs[int(pos)])

        idxs = idxs - pruned_set
        probs = probs[~pruned]
        assert len(idxs) == len(probs)
    else:
        pruned_fraction = 0.
        pruned_set = pr.BitMap()
    
    values = np.zeros_like(probs)
    for j,idx in enumerate(idxs):        
        ans = _solve_idx(idx, i)
        values[j,:] = ans

    expected_utils = (probs * values).sum(axis=-1)
    assert expected_utils.shape[0] == values.shape[0]
    pos = np.argmax(expected_utils)
    return Result(value=expected_utils[pos], index=idxs[int(pos)], pruned_fraction=pruned_fraction)

def efficient_nonmyopic_search(model : ProbabilityModel, *, time_horizon : int,  lookahead_limit : int, pruning_on : bool) -> Result:
    ''' lookahead_limit: 0 means no tree search, 1 
        time_horizon: how many moves into the future
    '''
    assert 1 <= lookahead_limit <= 2, 'implementation assumes at most 1 lookahead (pruning)'
    assert lookahead_limit <= time_horizon
    assert time_horizon > 0
    return _opt_expected_utility_helper(i=0, lookahead_limit=lookahead_limit, t=time_horizon, model=model, pruning_on=pruning_on)
