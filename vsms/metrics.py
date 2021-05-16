import numpy as np
import sklearn.metrics

def ndcg_rank_score(ytrue, ordered_idxs):
    '''
        wraps sklearn.metrics.ndcg_score to grade a ranking given as an ordered array of k indices,
        rather than as a score over the full dataset.
    '''
    ytrue = ytrue.astype('float')
    # create a score that is consistent with the ranking.
    ypred = np.zeros_like(ytrue)
    ypred[ordered_idxs] = np.arange(ordered_idxs.shape[0],0,-1)/ordered_idxs.shape[0]
    return sklearn.metrics.ndcg_score(ytrue.reshape(1,-1), y_score=ypred.reshape(1,-1), k=ordered_idxs.shape[0])

def test_score_sanity():
    ytrue = np.zeros(10000)
    randorder = np.random.permutation(10000)
    
    numpos = 100
    posidxs = randorder[:numpos]
    negidxs = randorder[numpos:]
    numneg = negidxs.shape[0]

    ytrue[posidxs] = 1.
    perfect_rank = np.argsort(-ytrue)
    bad_rank = np.argsort(ytrue)

    ## check score for a perfect result set is  1
    ## regardless of whether the rank is computed when k < numpos or k >> numpos
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos//2]),1.)
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos]),1.)
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos*2]),1.)    
    
        ## check score for no results is 0    
    assert np.isclose(ndcg_rank_score(ytrue,bad_rank[:-numpos]),0.)

  
    ## check score for same amount of results worsens if they are shifted    
    gr = perfect_rank[:numpos//2]
    br = bad_rank[:numpos//2]
    rank1 = np.concatenate([gr,br])
    rank2 = np.concatenate([br,gr])
    assert ndcg_rank_score(ytrue, rank1) > .5, 'half of entries being relevant, but first half'
    assert ndcg_rank_score(ytrue, rank2) < .5

def test_score_rare():
    n = 10000 # num items
    randorder = np.random.permutation(n)

    ## check in a case with only few positives
    for numpos in [0,1,2]:
        ytrue = np.zeros_like(randorder)
        posidxs = randorder[:numpos]
        negidxs = randorder[numpos:]
        numneg = negidxs.shape[0]
    
        ytrue[posidxs] = 1.
        perfect_rank = np.argsort(-ytrue)
        bad_rank = np.argsort(ytrue)
    
        scores = []
        k = 200
        for i in range(k):
            test_rank = bad_rank[:k].copy()
            if len(posidxs) > 0:
                test_rank[i] = posidxs[0]
            sc = ndcg_rank_score(ytrue,test_rank)
            scores.append(sc)
        scores = np.array(scores)
        if numpos == 0:
            assert np.isclose(scores ,0).all()
        else:
            assert (scores > 0 ).all()

test_score_sanity()
test_score_rare()