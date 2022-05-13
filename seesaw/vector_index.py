import ray
import annoy
import numpy as np
from .definitions import FS_CACHE
import pickle
import time


def build_annoy_idx(*, vecs, output_path, n_trees):
    start = time.time()
    t = annoy.AnnoyIndex(512, "dot")  # Length of item vector that will be indexed
    for i in range(len(vecs)):
        t.add_item(i, vecs[i])
    print(f"done adding items...{time.time() - start} sec.")
    t.build(n_trees=n_trees)  # 10 trees
    delta = time.time() - start
    print(f"done building...{delta} sec.")
    t.save(output_path)
    return delta


def build_nndescent_idx(vecs, output_path, n_trees):
    import pynndescent

    start = time.time()
    ret = pynndescent.NNDescent(
        vecs.copy(),
        metric="dot",
        n_neighbors=100,
        n_trees=n_trees,
        diversify_prob=0.5,
        pruning_degree_multiplier=2.0,
        low_memory=False,
    )
    print("first phase done...")
    ret.prepare()
    print("prepare done... writing output...", output_path)
    end = time.time()
    difftime = end - start
    pickle.dump(ret, file=open(output_path, "wb"))
    return difftime


class VectorIndex:
    def __init__(self, *, load_path, prefault=False):
        t = annoy.AnnoyIndex(512, "dot")
        self.vec_index = t
        load_path = FS_CACHE.get(load_path)
        t.load(load_path, prefault=prefault)
        print("done loading")

    def ready(self):
        return True

    def query(self, vector, top_k):
        assert vector.shape == (1, 512) or vector.shape == (512,)
        idxs, scores = self.vec_index.get_nns_by_vector(
            vector.reshape(-1), n=top_k, include_distances=True
        )
        return np.array(idxs), np.array(scores)
