from seesaw.metrics import *

def test_average_precision():  
    # perfect case
    AP = average_precision(np.array([0,1,2]), nseen=10, npositive=3)
    assert AP == 1.0
    
    # nothing found case
    AP = average_precision(np.array([]), nseen=10, npositive=3)
    assert AP == 0.
    
    # perfect case one elt
    AP = average_precision(np.array([0]), nseen=10, npositive=1)
    assert AP == 1.0
    
    # imperfect case: missing some
    AP_0 = average_precision(np.array([0,1,2]), nseen=10, npositive=10)
    assert AP_0 == 3./10
        
    # imperfect case: some false positives first
    AP_1 = average_precision(np.array([1,2,3]), nseen=10, npositive=3)
    assert AP_1 == (1./2 + 2./3 + 3./4)/3.
    
    # both kinds of imperfections:
    AP_01 = average_precision(np.array([1,2,3]), nseen=10, npositive=10)
    assert AP_01 == (1./2 + 2./3 + 3./4)/10.
    assert AP_01 < AP_0
    assert AP_01 < AP_1
    

def test_ndcg():
    ndcg = ndcg_score(np.array([0,1,2]), nseen=10, npositive=3)
    assert ndcg == 1.
    
    ndcg = ndcg_score(np.array([]), nseen=10, npositive=3)
    assert ndcg == 0.

    # perfect case one element
    ndcg = ndcg_score(np.array([0]), nseen=10, npositive=1)
    assert ndcg == 1.
    
    # imperfect case: missing some
    ndcg_0 = ndcg_score(np.array([0,1,2]), nseen=10, npositive=4)
    assert ndcg_0 < 1.

    # imperfect case: not first
    ndcg_1 = ndcg_score(np.array([1,2,3]), nseen=10, npositive=3)
    assert ndcg_1 < 1.
    
    # imperfect case: both 
    ndcg_01 = ndcg_score(np.array([1,2,3]), nseen=10, npositive=4)
    assert ndcg_01 < ndcg_0
    assert ndcg_01 < ndcg_1

    # unnormalized. check index 0 is handled properly
    dcg = dcg_score(np.array([0]), nseen=10, npositive=3)
    assert dcg == 1.
    
def test_time_to_kth():
    tt = time_to_kth(np.array([0]), k=1)
    assert tt == 1.

    tt = time_to_kth(np.array([]), k=1)
    assert tt == math.inf
        
    tt = time_to_kth(np.array([0]), k=2)
    assert tt == math.inf
    
    tt = time_to_kth(np.array([1,2,3]), k=1)
    assert tt == 2
    
    tt = time_to_kth(np.array([1,2,3]), k=2)
    assert tt == 3