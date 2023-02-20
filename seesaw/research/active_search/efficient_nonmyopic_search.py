
from seesaw.loops.LKNN_model import LKNNModel
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
        print(f'{idx=}, {ans=}')
        values[j,:] = ans

    expected_utils = (probs * (values + np.array([0,1]).reshape(1,-1))).sum(axis=-1)
    assert expected_utils.shape[0] == values.shape[0]
    pos = np.argmax(expected_utils)
    return Result(value=expected_utils[pos], index=idxs[int(pos)], pruned_fraction=pruned_fraction)

import math


def _top_sum(*, seen_idxs, numerators,  denominators, gamma, scores, neighbor_ids_sorted, N, K, D):
    """ returns the expected value after K steps for each index """
    
    top_kpd_ids = np.argsort(scores)[-(K+D):]
    top_scores = scores[top_kpd_ids]    

    ## first detect the over-writes to top k scores
    ## we do this by finding out if the insertion location element is equal to the value
    top_kpd_order_asc = np.argsort(top_kpd_ids)


    ## this sentinal above is so that searchsorted can position them correctly
    sentinel = numerators.shape[0]
    top_kpd_asc = np.concatenate([top_kpd_ids[top_kpd_order_asc], [sentinel]])
    top_score_by_kpd = top_scores[top_kpd_order_asc]
    
    insert_pos = np.searchsorted(top_kpd_asc, neighbor_ids_sorted)
    top_ids_in_position = top_kpd_asc[insert_pos]
    overwrites = (top_ids_in_position == neighbor_ids_sorted) # those which are equal to their insertion location
    iis, jjs = np.where(overwrites)
    jjs_in_topk = insert_pos[iis,jjs]
    
    assert (top_kpd_asc[jjs_in_topk] == neighbor_ids_sorted[iis,jjs]).all() , 'ids should match for there to be a conflict'
    # check the ids match. this was the goal of the above

    ### expand the top k scores by copying because we will over-write them
    top_score_by_kpd_rep = np.repeat(top_score_by_kpd, N).reshape(-1,N).T
    # top_id_rep = np.repeat(top_kpd_asc, N).reshape(-1,N).T
    
    ## make overwritten score -inf to self, and to overwritten elements so it will be ignored when sorting
    self_id = top_kpd_asc[:-1].reshape(1,-1) == node_ids
    top_score_by_kpd_rep[self_id] = -np.inf

    top_score_by_kpd_rep[iis, jjs_in_topk]  = -np.inf

    ## double check? 
    top_score_by_kpd_rep = top_score_by_kpd_rep[:,:-1] # remove sentinel inf

    def _compute_conditioned_scores(new_scores1):
        self_id = (neighbor_ids_sorted == node_ids)
        neighbor_scores1 = np.take(new_scores1, neighbor_ids_sorted)
        ## removes itself from topk
        neighbor_scores1[self_id] = - math.inf
        top_kp2d_scores = np.concatenate([top_score_by_kpd_rep, neighbor_scores1], axis=-1)
        #top_kp2d_ids = np.concatenate([top_id_rep, neighbor_ids_sorted], axis=-1)
    
        # now sort scores
        posns = np.argsort(top_kp2d_scores)
        top_k_scores = np.take_along_axis(top_kp2d_scores, posns[:,-K:], axis=1)
        
        assert (top_k_scores > -np.inf).all() # sanity check
        # top_k_ids = np.take_along_axis(top_kp2d_ids, posns[:,-K:], axis=1)

        ## sums 
        expected_scores1 = top_k_scores.sum(axis=1)
        return expected_scores1
    
    new_denom = denominators + 1
    new_scores1 = (numerators + 1)/new_denom
    new_scores0 = numerators/new_denom

    print(f'{np.nanmax(new_scores1)=} {np.nanmax(new_scores0)=}')

    assert (new_scores1 <= 1).all()
    assert ((new_scores0 >= 0) | (new_scores0 == -np.inf)).all()
    assert (new_scores0 <= new_scores1).all()
    
    expected_scores1 = _compute_conditioned_scores(new_scores1)
    expected_scores0 = _compute_conditioned_scores(new_scores0)
    
    ### need to compute scores p1*e1 + p0*e0
    ## the infinity scores (which have been cancelled) will become nan when added to plus infinity
    return scores*(1+expected_scores1) + (1-scores)*expected_scores0


def _opt_expected_utility_helper_lknn2(*, i : int,  lookahead_limit : int, t : int, model : LKNNModel, pruning_on : bool):
    assert i == 0
    assert lookahead_limit <=2
    assert t >= lookahead_limit

    ## first version
    deltas = model.matrix.indptr[1:] - model.matrix.indptr[:-1]
    assert (deltas == deltas[0]).all()
    D = deltas[0]

    neighbor_ids = model.matrix.indices.reshape(-1,D)
    neighbor_ids_sorted = np.sort(neighbor_ids)
    N = neighbor_ids_sorted.shape[0]


    numerators = model.numerators + model.gamma
    denominators = model.denominators + 1
    numerators[model.dataset.seen_indices] = -math.inf # will rank lowest
    scores = numerators/denominators

    if lookahead_limit == 2:
        expected_value =  _top_sum(seen_idxs=model.dataset.seen_indices, 
                                        numerators=numerators, denominators=denominators, 
                                        scores=scores,
                                        gamma=model.gamma,
                                        neighbor_ids_sorted=neighbor_ids_sorted, N=N, K=t-1, D=D)
        best_idx = np.nanargmax(expected_value)
        return Result(value=expected_value[best_idx], index=best_idx, pruned_fraction=0.)
    else:
        assert lookahead_limit == 1
        assert t == 1

        best_idx = np.nanargmax(scores)
        return Result(value=scores[best_idx], index=best_idx, pruned_fraction=0.)


        
    ## goal: get sums of the top_k remaining scores.
    ## note: when idx appears on both sides, old value should not count.

    



def efficient_nonmyopic_search(model : ProbabilityModel, *, time_horizon : int,  lookahead_limit : int, pruning_on : bool) -> Result:
    ''' lookahead_limit: 0 means no tree search, 1 
        time_horizon: how many moves into the future
    '''
    assert 1 <= lookahead_limit <= 2, 'implementation assumes at most 1 lookahead (pruning)'
    assert lookahead_limit <= time_horizon
    assert time_horizon > 0
    #return _opt_expected_utility_helper(i=0, lookahead_limit=lookahead_limit, t=time_horizon, model=model, pruning_on=pruning_on)
    return _opt_expected_utility_helper_lknn2(i=0, lookahead_limit=lookahead_limit, t=time_horizon, model=model, pruning_on=pruning_on)

