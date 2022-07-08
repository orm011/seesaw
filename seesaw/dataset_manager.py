from .memory_cache import CacheStub
from .query_interface import AccessMethod
import os
import numpy as np
import pandas as pd
import shutil
from .dataset import SeesawDatasetManager


class GlobalDataManager:
    global_cache: CacheStub

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.exists(root):
            print(f"creating new root folder at {root}")
            os.makedirs(root)

        self.root = root
        self.data_root = f"{root}/data/"
        self.model_root = f"{root}/models/"
        self.index_root = f"{root}/indices/"
        self.global_cache = CacheStub("actor#cache")

        paths = [self.data_root, self.model_root, self.index_root]
        for p in paths:
            os.makedirs(p, exist_ok=True)

    def list_datasets(self):
        return os.listdir(self.data_root)

    def list_indices(self):
        return []  # TODO: fix this

    def load_index(self, dataset_name, index_name) -> AccessMethod:
        index_path = f"{self.root}/data/{dataset_name}/indices/{index_name}"
        return AccessMethod.load(index_path)

    def get_dataset(self, dataset_name) -> SeesawDatasetManager:
        dataset_path = f"{self.root}/data/{dataset_name}"
        return SeesawDatasetManager(dataset_path, cache=self.global_cache)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.root})"
