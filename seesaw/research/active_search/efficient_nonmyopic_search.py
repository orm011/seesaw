
from seesaw.loops.LKNN_model import LKNNModel
from .common import ProbabilityModel, Result
import numpy as np
import pyroaring as pr

def _expected_utility_approx(t: int, model : ProbabilityModel, verbose=False):
    assert t > 0

    idxs, scores = model.top_k_remaining(top_k=t)
    if verbose:
        print(f'{model.dataset.seen_indices=}')
        print(f'{idxs=}\n{scores=}')
    next_idx = idxs[0]
    expected_u = scores.sum()
    return Result(value=expected_u, index=next_idx, pruned_fraction=None)

def _opt_expected_utility_helper(*, i : int,  lookahead_limit : int, t : int, model : ProbabilityModel, pruning_on : bool, verbose=False):
    '''l: lookahead exact horizon
       t: lookahead total horizon
       
       returns the expected utlity at horizon t for this batch, assuming an exact
       look-ahead of k <= t
    '''

    assert i >= 0
    assert i < lookahead_limit
    if (i == lookahead_limit - 1):
#        print(f'{t-i=}')
        return _expected_utility_approx(t - i, model, verbose)

    idxs = model.dataset.remaining_indices()
    p1 = model.predict_proba(idxs).reshape(-1,1)

    probs = np.concatenate([1-p1, p1], axis=-1)
    assert probs.shape[0] == p1.shape[0]
    assert probs.shape[1] == 2

    def _solve_idx(idx, i, verbose=False):
        if verbose:
            print(f'{idx} cond0')
        util0 = _opt_expected_utility_helper(i=i+1, lookahead_limit=lookahead_limit, t=t, model=model.condition(idx, 0), pruning_on=pruning_on, verbose=verbose)
        if verbose:
            print(f'\n{idx} cond1')
        util1 = _opt_expected_utility_helper(i=i+1, lookahead_limit=lookahead_limit, t=t, model=model.condition(idx, 1), pruning_on=pruning_on, verbose=verbose)
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

    expected_utils = (probs * (values + np.array([0,1]).reshape(1,-1))).sum(axis=-1)
    assert expected_utils.shape[0] == values.shape[0]
    pos = np.argmax(expected_utils)
    return Result(value=expected_utils[pos], index=idxs[int(pos)], pruned_fraction=pruned_fraction)

import math


def _top_sum(*, numerators,  denominators,  scores, neighbor_ids_sorted, N, K, D, debug=True):
    """ returns the expected value after K steps for each index """

    node_ids = np.arange(N).reshape(-1,1)
    top_kpd_ids = np.argsort(scores)[-(K+D):]
    top_scores = scores[top_kpd_ids]    

    ## first detect the over-writes to top k scores
    ## we do this by finding out if the insertion location element is equal to the value
    top_kpd_order_asc = np.argsort(top_kpd_ids)
    top_kpd_asc = top_kpd_ids[top_kpd_order_asc]
    top_score_by_kpd = top_scores[top_kpd_order_asc]


    ## 2. Detect any conflicting scores due to neighbor update. the neighbor update must win,
    ## therefore we set the old score to -inf. 
    ## we do this on a row by row basis
    ## this is shared regardless on the value we condition on, it only depends on neighbors that changed
    top_kpd_plus_sentinel = np.concatenate([top_kpd_asc,np.array([N])])
    insert_pos = np.searchsorted(top_kpd_asc, neighbor_ids_sorted)
    top_ids_in_position = top_kpd_plus_sentinel[insert_pos]
    overwrites = (top_ids_in_position == neighbor_ids_sorted) 
    iis, jjs = np.where(overwrites)
    jjs_in_topk = insert_pos[iis,jjs]

    if debug:
        assert (top_kpd_asc[jjs_in_topk] == neighbor_ids_sorted[iis,jjs]).all() , 'ids should match for there to be a conflict'

    ### expand the top k scores by copying because we will over-write them
    top_score_by_kpd_rep = np.repeat(top_score_by_kpd, N).reshape(-1,N).T
    top_id_rep = np.repeat(top_kpd_asc, N).reshape(-1,N).T
    
    ## make overwritten score -inf to self, and to overwritten elements so it will be ignored when sorting
    self_id = top_kpd_asc.reshape(1,-1) == node_ids
    top_score_by_kpd_rep[self_id] = -np.inf
    top_score_by_kpd_rep[iis, jjs_in_topk]  = -np.inf

    assert top_score_by_kpd_rep.shape == top_id_rep.shape, f'{top_score_by_kpd_rep.shape=} {top_id_rep.shape=}'

    def _compute_conditioned_scores(new_scores):
        self_id = (neighbor_ids_sorted == node_ids)
        neighbor_scores1 = np.take(new_scores, neighbor_ids_sorted)
        ## removes itself from topk
        neighbor_scores1[self_id] = - math.inf
        top_kp2d_scores = np.concatenate([top_score_by_kpd_rep, neighbor_scores1], axis=-1)
        top_kp2d_ids = np.concatenate([top_id_rep, neighbor_ids_sorted], axis=-1)
        assert top_kp2d_scores.shape == top_kp2d_ids.shape, f'{top_kp2d_scores.shape=} {top_kp2d_ids.shape=} {top_score_by_kpd_rep.shape=} {top_id_rep.shape=} {neighbor_ids_sorted.shape=}'
    
        # now sort scores
        order_asc = np.argsort(top_kp2d_scores)
        order_desc = np.fliplr(order_asc)
        order_desc = order_desc[:,:K] # keep only top K for each row

        top_k_scores = np.take_along_axis(top_kp2d_scores, order_desc, axis=1)

        if debug:
            top_k_ids = np.take_along_axis(top_kp2d_ids, order_desc, axis=1) # not needed unless debugging
            assert (top_k_ids != node_ids).all()

        assert (top_k_scores > -np.inf).all() # sanity check
        return top_k_scores.sum(axis=1)
    
    new_denom = denominators + 1
    scores_given0 = numerators/new_denom
    scores_given1 = (numerators + 1)/new_denom

    assert (scores_given1 <= 1).all()
    assert ((scores_given0 >= 0) | (scores_given0 == -np.inf)).all()
    assert (scores_given0 <= scores_given1).all()

    expected_scores1 = _compute_conditioned_scores(scores_given1)
    expected_scores0 = _compute_conditioned_scores(scores_given0)
    
    ## :NB: the infinity scores (which have been cancelled) will become nan when added to plus infinity
    final_scores = scores*(1+expected_scores1) + (1-scores)*expected_scores0
    return final_scores

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

    assert ((0 < model.gamma) & (model.gamma < 1)).all()
    assert (model.numerators <= model.denominators).all()

    numerators = model.numerators + model.gamma
    denominators = model.denominators + 1
    numerators[model.dataset.seen_indices] = -math.inf # will rank lowest
    scores = numerators/denominators

    assert (numerators <= denominators).all()

    if lookahead_limit == 2:
        expected_value =  _top_sum(numerators=numerators, denominators=denominators, 
                                    scores=scores,
                                        neighbor_ids_sorted=neighbor_ids_sorted, N=N, K=t-1, D=D)
        best_idx = np.nanargmax(expected_value)
        return Result(value=expected_value[best_idx], index=best_idx, pruned_fraction=0.)
    else:
        assert lookahead_limit == 1, lookahead_limit
        # assert t == 1, t

        best_idx = np.nanargmax(scores)
        return Result(value=scores[best_idx], index=best_idx, pruned_fraction=0.)


def efficient_nonmyopic_search(model : ProbabilityModel, *, reward_horizon : int,  lookahead_limit : int, pruning_on : bool, implementation : str) -> Result:
    ''' lookahead_limit: 0 means no tree search, 1 
        time_horizon: how many moves into the future
    '''
    assert reward_horizon > 0
    assert 1 <= lookahead_limit <= 2, 'implementation assumes at most 1 lookahead (pruning)'
    assert lookahead_limit <= reward_horizon

    if implementation == 'vectorized':
        return _opt_expected_utility_helper_lknn2(i=0, lookahead_limit=lookahead_limit, t=reward_horizon, model=model, pruning_on=pruning_on)
    elif implementation == 'loop':
        return _opt_expected_utility_helper(i=0, lookahead_limit=lookahead_limit, t=reward_horizon, model=model, pruning_on=pruning_on)


