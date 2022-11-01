import torch
import torch.nn.functional as F

def ref_signed_inversions(target, *, scores, margin):
    ''' computes the inversions (number of inversions) for the given scores,
        with respect to the given target.

        Target can include many distinct values, or repeated ones (eg when therea are only two relevances)
        
        counts all pairs i,j such that the following is violated:

            if target[i] < target[j] then score[i] + margin < score[j]
            and
            if target[i] > target[j] then scores[i] - margin > score[j]        

        implies that if target [i ] = target[j], any value of scores is allowed,
        but if target[i] != target[j], then the values must be different in the right direction.
        When margin is non-zero they must be different by at least the margin.
    '''
    assert target.shape == scores.shape
    target_diff = (target.reshape(-1,1) - target.reshape(1,-1)).sign() # want -1 or 1 or 0
    score_diff = scores.reshape(-1,1) - scores.reshape(1,-1) - margin*target_diff
    
    # if target_diff is 1, then decrease the difference value by the margin 
    # if target_diff is -1, then increase the difference toward 0
    # if target_diff is 0, means target value is equal, so value differences are the same.

    negative_inversion = (target_diff < 0) & (score_diff >= 0) 
    positive_inversion =  (target_diff > 0) & (score_diff <= 0) # equal is needed for case where scores are equal but target are not
 
    all_pairs = positive_inversion.float() - negative_inversion.float()
    return all_pairs


def ref_pairwise_rank_loss(target, *, scores, margin, aggregate='none'):
    ''' target is the right score (can be binary or not)
        scores are an output
        
        y_ij = (target[j] - target[i]).sign()
        s_ij = score[j] - score[i]
        
        hinge_loss = max(0, margin - y_ij * s_ij) # s_ij should be same sign as y_ij, s_ij and there should be a margin betwween them.
          #( if y_ij is 0, then s_ij is ignored)
    '''
    assert target.shape == scores.shape
    target_ij = (target.reshape(-1,1) - target.reshape(1,-1)).sign() # want -1 or 1 or 0
    score_ij = (scores.reshape(-1,1) - scores.reshape(1,-1))
        
    loss_ij = torch.clamp(margin -  target_ij * score_ij, min=0)
    
    # remove constant term for elements with identical targets, as it does not contribute to gradient.
    loss_ij -= margin*(target_ij == 0).float()

    if aggregate == 'none':
        return loss_ij
    elif aggregate == 'sum':
        return loss_ij.div(2).sum()
    else:
        assert False

    return loss_ij


def ref_pairwise_rank_loss_gradient(target, *, scores, margin):
    ''' reference implementation of gradient for pairwise rank loss.
        use autograd to compute the gradient.
    '''
    scores = scores.clone().detach().requires_grad_(True)
    loss = ref_pairwise_rank_loss(target, scores=scores, margin=margin, aggregate='sum')

    loss.backward()
    return scores.grad.clone()


def _quick_pairwise_gradient_sorted(sorted_targets, scores):
    ## assumes input targets,scores are sorted in lexicographic order

    # initially we just did:
    # _, final_indices = torch.sort(scores, stable=True)
    ## but when targets are unequal but scores are equal, in order to encourage divergence, 
    ## and to match reference, gradient needs to be non zero (boundary case)

    ## Implies the sorting of scores at this stage should not be stable, but anti-stable.
    ## where equal values get permuted in reverse, we can achieve this by
    ## sorting scores in lexicographic score, -target order as opposed to only score.
    _, _, final_indices = lexicographic_sort(scores, -sorted_targets, return_indices=True)
    _, reverse_indices = torch.sort(final_indices)

    ## reverse_indices is 0 based rank of current scores to go in the list (scatter indices)
    ## net_reversals is the the number of positions they need to change in relative terms, and 
    # it is equal to the gradient wrt hinge loss (with margin 0)
    net_reversals = reverse_indices - torch.arange(reverse_indices.shape[0])
    return net_reversals

def lexicographic_sort(a, b, return_indices=False):
    #sorts (a,b)
    b2, indices1 = torch.sort(b, stable=True)
    a2 = a[indices1]
    
    a3, indices2 = torch.sort(a2, stable=True)
    b3 = b2[indices2]
    indices_all = indices1[indices2]
    return a3, b3, indices_all

def quick_pairwise_gradient_zero_margin(target, *, scores):
    ''' computes the gradient of the pair-wise rank loss at the given inputs
        but in linear space and nlog time, by using a few sort passes.

        for now can only do margin=0.
    '''
    starget, sscores, sindex = lexicographic_sort(target, scores)
    _, invsindex = torch.sort(sindex)
    grads = _quick_pairwise_gradient_sorted(starget, sscores)
    ## now return gradient in the input order
    return grads[invsindex].float()


class _CheapPairwiseRankingLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target, scores):
        grads = quick_pairwise_gradient_zero_margin(target, scores=scores)
        ctx.save_for_backward(grads)
        ## not really margin loss, but net inversions  still informative
        return grads.abs() # vector 

    @staticmethod
    def backward(ctx, grad_output): # output is vector
        (grads,) = ctx.saved_tensors

        ## grad output propagates any scaling done after
        grad_target = None
        grad_scores = grads*grad_output

        return grad_target, grad_scores

def cheap_pairwise_rank_loss(target, *, scores):
    return _CheapPairwiseRankingLoss.apply(target, scores)