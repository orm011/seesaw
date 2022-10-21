from .memory_cache import CacheStub
from .query_interface import AccessMethod
import os
from .dataset import SeesawDataset
from .basic_types import IndexSpec, SessionParams

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

    def get_dataset(self, dataset_name) -> SeesawDataset:
        dataset_path = f"{self.root}/data/{dataset_name}"
        return SeesawDataset(dataset_path, cache=self.global_cache)
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.root})"
