from seesaw.loops.LKNN_model import LKNNModel
from seesaw.research.active_search.common import Dataset
import numpy as np
import scipy.sparse as sp


def make_ring_graph_matrix_model():
    dataset = Dataset.from_vectors(np.random.random((5,10)))

    mat = np.zeros((5,5))
    for i in range(5):
        mat[i,(i+1) % 5] = 1

    sym = (mat + mat.T)
    matrix=  sp.csr_array(sym)
    return LKNNModel.from_dataset(dataset, weight_matrix=matrix, gamma=.5)


def test_predictions():
    model = make_ring_graph_matrix_model()
    points = np.array([0,1,2,3,4])

    probs = model.predict_proba(points)
    assert np.isclose(probs ,.5).all()

    ## check update
    probs2 = model.condition(2, 1).predict_proba(points)
    assert np.isclose(probs2, np.array([.5, .75, 1., .75, .5])).all()
    
    ## check no mutation
    probs = model.predict_proba(points)
    assert np.isclose(probs,.5).all() # check no update

    ## check opposite update
    probs4  = model.condition(2,0).predict_proba(points)
    assert np.isclose(probs4, np.array([.5, .25, 0., .25, .5])).all()


def test_bound():
    model = make_ring_graph_matrix_model()
    pbound1 = model.probability_bound(1)
    assert .75 <= pbound1

    pbound2 = model.probability_bound(2)
    assert 2.5/3 <= pbound2








