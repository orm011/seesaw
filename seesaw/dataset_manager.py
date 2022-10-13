from .memory_cache import CacheStub
from .query_interface import AccessMethod
import os
from .dataset import SeesawDatasetManager
from .basic_types import IndexSpec

class GlobalDataManager:
    global_cache: CacheStub

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.exists(root):
            print(
                f"No existing root found at {root}. Creating new root folder at {root}"
            )
            os.makedirs(root)

        self.root = root
        self.data_root = f"{root}/data/"
        self.model_root = f"{root}/models/"
        self.global_cache = CacheStub("actor#cache")

        paths = [self.data_root, self.model_root]
        for p in paths:
            os.makedirs(p, exist_ok=True)

    def list_datasets(self):
        return os.listdir(self.data_root)

    def list_indices(self, dataset):
        return os.listdir(f"{self.data_root}/{dataset}/indices/")

    def load_index(self, dataset_name, index_name, *, options) -> AccessMethod:
        index_path = f"{self.root}/data/{dataset_name}/indices/{index_name}"
        return AccessMethod.load(index_path, options=options)

    def get_dataset(self, dataset_name) -> SeesawDatasetManager:
        dataset_path = f"{self.root}/data/{dataset_name}"
        return SeesawDatasetManager(dataset_path, cache=self.global_cache)

    def _get_knng_path(self, ispec: IndexSpec):
        base =   f'{self.root}/data/{ispec.d_name}/indices/{ispec.i_name}'

        if ispec.c_name is None or ispec.d_name != 'lvis': # HACK. lvis treat lvis separately since each category has a subset
            return f'{base}/knn_graph'
        else:
            return f'{base}/subsets/{ispec.c_name}'
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.root})"
