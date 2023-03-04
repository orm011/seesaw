import math
import numpy as np


def average_reciprocal_gap(*args, **kwargs):
    return average_precision(*args, **kwargs, average_reciprocal_gap=True)

def average_precision(hit_indices, *, npositive, max_results=None, average_reciprocal_gap=False):
    ### average the precision at every point a new instance is seen
    ### for those unseen instances, they score zero.
    ## for consistency across experiments we ignore results after max_results (treated as not found)
    assert npositive > 0
    
    if max_results is None:
        max_results = npositive        
    max_results = min(npositive,max_results)
    
    hit_indices = hit_indices[:max_results]
    ranks = hit_indices + 1
    
    denominators = np.ones(max_results) * np.inf
    if average_reciprocal_gap:
        ranks2 = np.concatenate((np.zeros(1), ranks))
        gaps = ranks2[1:] - ranks2[:-1]

        numerator = 1.
        denominators[:hit_indices.shape[0]] = gaps
    else:
        numerator = np.arange(denominators.shape[0]) + 1
        denominators[:hit_indices.shape[0]] = ranks
        
    ratio = numerator/denominators
    # print(numerator, denominators, ratio)
    return np.mean(ratio)

# average_precision(np.array([0,1,2]), npositive=4, max_results=3) == 1.

def dcg_score(hit_indices):
    # weights following https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/metrics/_ranking.py#L1239
    weights = 1.0 / np.log2(hit_indices + 2)
    dcg_score = weights.sum()
    return dcg_score


def rank_of_kth(hit_indices, *, ntotal, k):
    if k > ntotal: # not valid for this
        return None

    if hit_indices.shape[0] < k:
        return math.inf
    else:
        return hit_indices[k - 1] + 1

def rank_kth(hit_indices, *, ntotal, ks): # batch
    ans = np.ones_like(ks, dtype=float)
    ans[ks > hit_indices.shape[0]] = np.inf # didn't find it
    ans[ks > ntotal] = np.nan # not applicable
    ans[ks <= hit_indices.shape[0]] = hit_indices[ks[ks <= hit_indices.shape[0]] - 1] + 1
    return ans


def best_possible_hits(nseen, npositive):
    if npositive < nseen:
        return np.arange(npositive)
    else:
        return np.arange(nseen)


def ndcg_score(hit_indices, *, nseen, npositive):
    best_hits = best_possible_hits(nseen, npositive)
    return dcg_score(hit_indices) / dcg_score(best_hits)


def normalizedAP(hit_indices, *, nseen, npositive, max_results=None):
    best_hits = best_possible_hits(nseen, npositive)
    best_AP = average_precision(
        best_hits,  npositive=npositive, max_results=max_results
    )
    return (
        average_precision(
            hit_indices, npositive=npositive, max_results=max_results
        )
        / best_AP
    )


def batch_metrics(hit_indices, *, nseen, npositive, batch_size):
    batch_no = nseen // batch_size
    nfirst_batch = batch_no[0]
    nfirst = nseen[0]

    if (batch_no > batch_no[0]).any():
        gtpos = np.where(batch_no > batch_no[0])[0]
        assert gtpos.shape[0] > 0
        first_after_feedback = gtpos[0]
        nfirst2second_batch = batch_no[first_after_feedback] - nfirst_batch
        nfirst2second = nseen[first_after_feedback] - nfirst
    else:
        # only one batch contains all positives (or maybe no positives at all)
        # this metric is not well defined.
        nfirst2second = np.nan
        nfirst2second_batch = np.nan
    # TODO: reimplement this later


def compute_metrics(*, hit_indices, batch_size, nseen, ntotal, max_results):
    AP = average_precision(
        hit_indices, npositive=ntotal, max_results=max_results
    )
    #average_reciprocal = average_reciprocal_gap(hit_indices, npositive=ntotal, max_results=max_results)

    # nAP = normalizedAP(
    #     hit_indices, nseen=nseen, npositive=ntotal, max_results=max_results
    # )
    ndcg = ndcg_score(hit_indices, nseen=nseen, npositive=ntotal)
    rank_arr = rank_kth(hit_indices, ntotal=ntotal, ks=np.array([1,2,3,10]))
    rank_first, rank_second, rank_third, rank_tenth = rank_arr
    # rank_of_kth(hit_indices, ntotal=ntotal, k=1)
    # rank_second = rank_of_kth(hit_indices, ntotal=ntotal, k=2)
    # rank_third = rank_of_kth(hit_indices, ntotal=ntotal, k=3)
    # rank_tenth = rank_of_kth(hit_indices, ntotal=ntotal, k=10)

    nfound = hit_indices.shape[0]

    # only return things not given as input
    return dict(
        nfound=nfound,
        ndcg_score=ndcg,
        average_precision=AP,
#        average_reciprocal_gap=average_reciprocal,
#        nAP=nAP,
        rank_first=rank_first,
        reciprocal_rank=1./rank_first,
        rank_second=rank_second,
        rank_third=rank_third,
        rank_tenth=rank_tenth,
    )
