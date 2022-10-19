import numpy as np
import pyroaring as pr
import importlib
import json
from ..definitions import resolve_path


def get_constructor(cons_name: str):
    pieces = cons_name.split(".", maxsplit=-1)
    index_mod = importlib.import_module(".".join(pieces[:-1]))
    constructor = getattr(index_mod, pieces[-1])
    return constructor


class AccessMethod:
    path : str

    def string2vec(self, string: str) -> np.ndarray:
        raise NotImplementedError("implement me")

    def get_knng(self, path=None):
        from seesaw.research.knn_methods import KNNGraph
        if path is None:
            path = ''
            
        knng = KNNGraph.from_file(f'{self.path}/knn_graph/{path}')
        return knng

    def query(
        self, *, vector: np.ndarray, topk: int, exclude: pr.BitMap = None
    ) -> np.ndarray:
        raise NotImplementedError("implement me")

    def new_query(self):
        raise NotImplementedError("implement me")

    def subset(self, indices: pr.BitMap):
        raise NotImplementedError("implement me")

    @staticmethod
    def from_path(index_path: str, **options):
        raise NotImplementedError("implement me")

    @staticmethod
    def load(index_path: str, *, options : dict = None):
        index_path = resolve_path(index_path)
        meta = json.load(open(f"{index_path}/info.json", "r"))
        constructor_name = meta["constructor"]
        c = get_constructor(constructor_name)
        if options is None:
            options = {}
        
        return c.from_path(index_path, **options)
