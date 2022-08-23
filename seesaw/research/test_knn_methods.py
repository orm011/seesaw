from .knn_methods import SimpleKNNRanker, TensorArray
import numpy as np
import pandas as pd

def test_knn_ranker():
    knns = np.array([[1, 2], 
            [0, 2],
            [1, 0],
            [1, 2]]  # point[3] is further away from point[0]
            )
    
    dists = np.array([[.1, .3],
            [.1, .2],
            [.2, .3],
            [.4, .5]
            ])

    df = pd.DataFrame({'positions':TensorArray(knns), 'distances':TensorArray(dists)})

    scores = np.array([.9, .52, .51, .5])/2
    # should start with 0, then upon finding it is 0, should lower everyting else, jump to 3.
    knnr = SimpleKNNRanker(df, init_scores=scores)
    knnr2 = SimpleKNNRanker(df, init_scores=scores)


    base_scores = knnr.current_scores()
    assert (base_scores < 1.).all()
    assert (base_scores > 0.).all()
    
    ## test order is correct
    x, scores = knnr.top_k(4)
    assert (x == np.array([0,1,2,3])).all()

    x_, _ = knnr.top_k(2)
    assert (x_ == np.array([0,1])).all()


    ## test negative update
    knnr.update(idx=0, label=0)
    ## check scores decreased
    neg_update = knnr.current_scores()
    assert (base_scores[[0,1,2]] > neg_update[[0,1,2]]).all(), 'all scores should have decreased or stayed the same'
    assert base_scores[3] == neg_update[3], 'all scores should have decreased or stayed the same'
    # check ranking changes
    x2, scores2 = knnr.top_k(4, unlabeled_only=False)
    assert scores2[-1] == 0
    assert (x2 == np.array([3, 1, 2, 0])).all()
    # test 
    x3, scores3 = knnr.top_k(4)
    assert x3.shape[0] == 3
    assert (x3 == np.array([3, 1, 2])).all()
    assert (scores2[:-1] == scores3).all()
 
    # update to same label should change nothing
    knnr.update(idx=0, label=0) # nothing should happen
    re_update_scores = knnr.current_scores()
    assert (re_update_scores == neg_update).all()

    # update to different label
    knnr.update(idx=0, label=1) # change labels
    knnr2.update(idx=0, label=1) # first update
    pos_update = knnr.current_scores()
    pos_update_init = knnr2.current_scores()
    assert (pos_update == pos_update_init).all(), 'scores should match after update, as if previous update never happened'

    # check score change
    assert (base_scores[[0,1,2]] < pos_update[[0,1,2]]).all(), 'all scores connected to 0 should have increased'
