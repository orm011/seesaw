from seesaw.loops.LKNN_model import LazyTopK, Dataset
import numpy as np

def test_basic():
    dataset = Dataset.from_vectors(np.random.randn(5,3))
   
    scores = np.array([.5, .4, .3, .2, .1])
    ltk = LazyTopK(dataset, desc_idxs=np.arange(5), desc_scores=scores, 
                    desc_changed_idxs=np.array([3,2]), desc_changed_scores=np.array([.6, .0]))

    top_idxs, top_scores = ltk.top_k_remaining(k=2)
    assert np.equal(top_idxs, np.array([3, 0])).all()
    assert np.isclose(top_scores, np.array([.6, .5])).all()


    ltk = LazyTopK(dataset.with_label(0, 1), desc_idxs=np.arange(5), desc_scores=scores, 
                    desc_changed_idxs=np.array([3,2]), desc_changed_scores=np.array([.6, .0]))

    top_idxs, top_scores = ltk.top_k_remaining(k=2)
    assert np.equal(top_idxs, np.array([3, 1])).all()
    assert np.isclose(top_scores, np.array([.6, .4])).all()


    ltk = LazyTopK(dataset.with_label(0, 1).with_label(3,1), desc_idxs=np.arange(5), desc_scores=scores, 
                    desc_changed_idxs=np.array([3,2]), desc_changed_scores=np.array([.6, .0]))

    top_idxs, top_scores = ltk.top_k_remaining(k=2)
    assert np.equal(top_idxs, np.array([1,4])).all()
    assert np.isclose(top_scores, np.array([.4, .1])).all()


def test_iter_desc_scores():
    dataset = Dataset.from_vectors(np.random.randn(5,3))

    scores = np.array([.5, .4, .3, .2, .1])
    ltk = LazyTopK(dataset, desc_idxs=np.arange(5), desc_scores=scores, 
                    desc_changed_idxs=np.array([3,2]), desc_changed_scores=np.array([.6, .0]))

    ret = list(ltk._iter_desc_scores())
    assert len(ret) == 3

    ret2 = list(ltk._iter_desc_scores())
    assert len(ret2) == 3


    iter1 = ltk._iter_desc_scores()
    ret3 = []
    try:
        while True:
            ret3.append(next(iter1))
    except StopIteration:
        pass

    assert len(ret3) == 3


def test_full():
    dataset = Dataset.from_vectors(np.random.randn(5,3))

    scores = np.array([.5, .4, .3, .2, .1])
    ltk = LazyTopK(dataset, desc_idxs=np.arange(5), desc_scores=scores, 
                    desc_changed_idxs=np.array([3,2]), desc_changed_scores=np.array([.6, .0]))

    top_idxs, top_scores = ltk.top_k_remaining(k=6)
    assert np.equal(top_idxs, np.array([3, 0, 1, 4, 2]) ).all()
    assert np.isclose(top_scores, np.array([.6, .5, .4, .1, 0])).all()

    top_idxs, top_scores = ltk.top_k_remaining(k=6)
    assert np.equal(top_idxs, np.array([3, 0, 1, 4, 2]) ).all()
    assert np.isclose(top_scores, np.array([.6, .5, .4, .1, 0])).all()

    top_idxs, top_scores = ltk.top_k_remaining(k=1)
    assert np.equal(top_idxs, np.array([3]) ).all()
    assert np.isclose(top_scores, np.array([.6])).all()

    top_idxs, top_scores = ltk.top_k_remaining(k=10)
    assert np.equal(top_idxs, np.array([3, 0, 1, 4, 2]) ).all()
    assert np.isclose(top_scores, np.array([.6, .5, .4, .1, 0])).all()