import shutil
import ray
import annoy
import numpy as np
from .definitions import DATA_CACHE_DIR, parallel_copy


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
    def __init__(self, *, base_dir, load_path, copy_to_tmpdir: bool, prefault=False):
        t = annoy.AnnoyIndex(512, "dot")
        self.vec_index = t
        if copy_to_tmpdir:
            print("cacheing first", base_dir, DATA_CACHE_DIR, load_path)
            actual_load_path = parallel_copy(
                base_dir=base_dir, cache_dir=DATA_CACHE_DIR, rel_path=load_path
            )
        else:
            print("loading directly")
            actual_load_path = f"{base_dir}/{load_path}"

        t.load(actual_load_path, prefault=prefault)
        print("done loading")

    def ready(self):
        return True

    def query(self, vector, top_k):
        assert vector.shape == (1, 512) or vector.shape == (512,)
        idxs, scores = self.vec_index.get_nns_by_vector(
            vector.reshape(-1), n=top_k, include_distances=True
        )
        return np.array(idxs), np.array(scores)


RemoteVectorIndex = ray.remote(VectorIndex)


class IndexWrapper:
    def __init__(self, index_actor: RemoteVectorIndex):
        self.index_actor = index_actor

    def query(self, vector, top_k):
        h = self.index_actor.query.remote(vector, top_k)
        return ray.get(h)
