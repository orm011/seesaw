import math
import numpy as np

def average_precision(hit_indices, *, nseen, npositive):
    ### average the precision at every point a new instance is seen
    ### for those unseen instances, they score zero.
    assert npositive > 0
    assert nseen > 0
    assert (hit_indices < nseen).all()
    
    hpos_full = np.ones(npositive)*np.inf
    hpos_full[:hit_indices.shape[0]] = hit_indices
    total_seen_count = hpos_full + 1
    
    true_positive_count = np.arange(npositive) + 1
    precisions = true_positive_count/total_seen_count
    AP = np.mean(precisions)
    
    return AP

def dcg_score(hit_indices):
    # weights following https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/metrics/_ranking.py#L1239
    weights = 1./np.log2(hit_indices+2)
    dcg_score = weights.sum()
    return dcg_score
    
def time_to_kth(hit_indices, *, k):
    if hit_indices.shape[0] < k:
        return math.inf
    else:
        return hit_indices[k-1]+1
    
def best_possible_hits(nseen, npositive):
    if npositive < nseen:
        return np.arange(npositive)
    else:
        return np.arange(nseen)
    
def ndcg_score(hit_indices, *, nseen, npositive):
    best_hits = best_possible_hits(nseen, npositive)
    return dcg_score(hit_indices)/dcg_score(best_hits)

def normalizedAP(hit_indices, *, nseen, npositive):
    best_hits = best_possible_hits(nseen, npositive)
    best_AP = average_precision(best_hits, nseen=nseen, npositive=npositive)
    return average_precision(hit_indices, nseen=nseen, npositive=npositive)/best_AP


def batch_metrics(hit_indices, *, nseen, npositive, batch_size):
    batch_no = nseen//batch_size
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
    #TODO: reimplement this later

def compute_metrics(*, hit_indices, batch_size, nseen, ntotal):
    AP = average_precision(hit_indices, nseen=nseen, npositive=ntotal)
    nAP = normalizedAP(hit_indices, nseen=nseen, npositive=ntotal)
    ndcg = ndcg_score(hit_indices, nseen=nseen, npositive=ntotal)
    nfirst = time_to_kth(hit_indices, k=1)
    nfound = hit_indices.shape[0]

    return dict(nfound=nfound,
                ndcg_score=ndcg,
                AP=AP,
                nAP=nAP,
                nseen=nseen,
                nfirst=nfirst,
                ntotal=ntotal,
                reciprocal_rank=1./nfirst)