
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
            print(f'{idx} cond 0')
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
        #print(f'{idx=}, {ans=}')
        values[j,:] = ans

    expected_utils = (probs * (values + np.array([0,1]).reshape(1,-1))).sum(axis=-1)
    assert expected_utils.shape[0] == values.shape[0]
    pos = np.argmax(expected_utils)
    index = idxs[int(pos)]

    assert 30523 in idxs
    assert 45306 in idxs
    pos30523 = idxs.rank(30523) - 1
    pos45306 = idxs.rank(45306) - 1

    print(f'{model.gamma[30523]=}\n{model.gamma[45306]=}')
    #print(f'{_solve_idx(index, i)=}, {probs[pos]=}')
    print(f'{_solve_idx(45306, i, verbose=True)=}\n{probs[pos45306]=}')
    print(f'{_solve_idx(30523, i, verbose=True)=}\n{probs[pos30523]=}')


    return Result(value=expected_utils[pos], index=idxs[int(pos)], pruned_fraction=pruned_fraction)

import math


def _top_sum(*, numerators,  denominators,  scores, neighbor_ids_sorted, N, K, D):
    """ returns the expected value after K steps for each index """
    
    top_kpd_ids = np.argsort(scores)[-(K+D):]
    top_scores = scores[top_kpd_ids]    

    ## first detect the over-writes to top k scores
    ## we do this by finding out if the insertion location element is equal to the value
    top_kpd_order_asc = np.argsort(top_kpd_ids)

    node_ids = np.arange(N).reshape(-1,1)

    ## this sentinal above is so that searchsorted can position them correctly
    sentinel = numerators.shape[0] ## TODO: this is just N.
    top_kpd_asc = np.concatenate([top_kpd_ids[top_kpd_order_asc], [sentinel]])
    top_score_by_kpd = top_scores[top_kpd_order_asc]

    ##TODO: keep top_kpd_asc the same as the other one, and have one with the sentinel just for the stuff below.

    print(f'{top_kpd_asc[:-1]=}\n{top_score_by_kpd=}')

    
    insert_pos = np.searchsorted(top_kpd_asc, neighbor_ids_sorted)
    top_ids_in_position = top_kpd_asc[insert_pos]
    overwrites = (top_ids_in_position == neighbor_ids_sorted) 
    iis, jjs = np.where(overwrites)
    jjs_in_topk = insert_pos[iis,jjs]
    
    assert (top_kpd_asc[jjs_in_topk] == neighbor_ids_sorted[iis,jjs]).all() , 'ids should match for there to be a conflict'
    # check the ids match. this was the goal of the above


    print(f'{top_score_by_kpd.shape=} {top_kpd_asc.shape=}')
    ### expand the top k scores by copying because we will over-write them
    top_score_by_kpd_rep = np.repeat(top_score_by_kpd, N).reshape(-1,N).T
    top_id_rep = np.repeat(top_kpd_asc[:-1], N).reshape(-1,N).T
    
    ## make overwritten score -inf to self, and to overwritten elements so it will be ignored when sorting
    self_id = top_kpd_asc[:-1].reshape(1,-1) == node_ids
    top_score_by_kpd_rep[self_id] = -np.inf


    print(f'{top_score_by_kpd_rep[30523]=}\n{top_score_by_kpd_rep[45306]=}')

    top_score_by_kpd_rep[iis, jjs_in_topk]  = -np.inf

    assert top_score_by_kpd_rep.shape == top_id_rep.shape, f'{top_score_by_kpd_rep.shape=} {top_id_rep.shape=}'
    ##TODO: check agreement now.
    ##TODO: which assertions should we put in place here? 


    def _compute_conditioned_scores(new_scores1):
        self_id = (neighbor_ids_sorted == node_ids)
        neighbor_scores1 = np.take(new_scores1, neighbor_ids_sorted)
        ## removes itself from topk
        neighbor_scores1[self_id] = - math.inf
        top_kp2d_scores = np.concatenate([top_score_by_kpd_rep, neighbor_scores1], axis=-1)
        top_kp2d_ids = np.concatenate([top_id_rep, neighbor_ids_sorted], axis=-1)


        assert top_kp2d_scores.shape == top_kp2d_ids.shape, f'{top_kp2d_scores.shape=} {top_kp2d_ids.shape=} {top_score_by_kpd_rep.shape=} {top_id_rep.shape=} {neighbor_ids_sorted.shape=}'
        top_kp2d_isnew = np.concatenate([np.zeros(top_id_rep.shape[1]), np.ones(neighbor_ids_sorted.shape[1])], axis=-1)

#        print(f'{top_kp2d_scores[]=}\n{top_kp2d_ids=}')
    
        # now sort scores
        order_asc = np.argsort(top_kp2d_scores)
        order_desc = np.fliplr(order_asc)
        topKidxs = np.take_along_axis(top_kp2d_ids, order_desc, axis=1)
        top_k_scores = np.take_along_axis(top_kp2d_scores, order_desc, axis=1)

        print(f'{topKidxs[30523]=}\n{top_k_scores[30523]=}')
        print('\n')
        print(f'{topKidxs[45306]=}\n{top_k_scores[45306]=}')

        top_k_scores = top_k_scores[:,:K]
        assert (top_k_scores > -np.inf).all() # sanity check
        # top_k_ids = np.take_along_axis(top_kp2d_ids, posns[:,-K:], axis=1)
        expected_scores1 = top_k_scores.sum(axis=1)
        return expected_scores1
    
    new_denom = denominators + 1
    new_scores0 = numerators/new_denom
    new_scores1 = (numerators + 1)/new_denom

    assert (new_scores1 <= 1).all()
    assert ((new_scores0 >= 0) | (new_scores0 == -np.inf)).all()
    assert (new_scores0 <= new_scores1).all()

    print('condition on 1')    
    expected_scores1 = _compute_conditioned_scores(new_scores1)
    print('\n------\n')
    print('condition on 0')
    expected_scores0 = _compute_conditioned_scores(new_scores0)
    
    ### need to compute scores p1*e1 + p0*e0
    ## the infinity scores (which have been cancelled) will become nan when added to plus infinity
    final_scores = scores*(1+expected_scores1) + (1-scores)*expected_scores0
    index = np.nanargmax(final_scores)
    print(f'{expected_scores0[30523]=} {expected_scores1[30523]=}, {scores[30523]=} {(1-scores)[30523]=}, {neighbor_ids_sorted[index]=}')
    print(f'{expected_scores0[index]=} {expected_scores1[index]=}, {scores[index]=} {(1-scores)[index]=}, {neighbor_ids_sorted[index]=}')

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

    ### why are scores different?

    assert (numerators <= denominators).all()

    if lookahead_limit == 2:
        expected_value =  _top_sum(numerators=numerators, denominators=denominators, 
                                    scores=scores,
                                        neighbor_ids_sorted=neighbor_ids_sorted, N=N, K=t-1, D=D)
        best_idx = np.nanargmax(expected_value)
        print(f'{expected_value[30523]=} {numerators[30523]=} {denominators[30523]=}')
        print(f'{model.gamma[30523]=} {model.gamma[45306]=}')

        print(f'{scores[30523]=} {scores[45306]=}')

        return Result(value=expected_value[best_idx], index=best_idx, pruned_fraction=0.)
    else:
        assert lookahead_limit == 1, lookahead_limit
        # assert t == 1, t

        best_idx = np.nanargmax(scores)
        return Result(value=scores[best_idx], index=best_idx, pruned_fraction=0.)


def efficient_nonmyopic_search(model : ProbabilityModel, *, time_horizon : int,  lookahead_limit : int, pruning_on : bool, implementation : str) -> Result:
    ''' lookahead_limit: 0 means no tree search, 1 
        time_horizon: how many moves into the future
    '''
    assert 1 <= lookahead_limit <= 2, 'implementation assumes at most 1 lookahead (pruning)'
    assert lookahead_limit <= time_horizon
    assert time_horizon > 0

    print(f'{time_horizon=} {lookahead_limit=}')
    if implementation == 'vectorized':
        return _opt_expected_utility_helper_lknn2(i=0, lookahead_limit=lookahead_limit, t=time_horizon, model=model, pruning_on=pruning_on)
    elif implementation == 'loop':
        return _opt_expected_utility_helper(i=0, lookahead_limit=lookahead_limit, t=time_horizon, model=model, pruning_on=pruning_on)


