import numpy as np
from ..dataset_manager import GlobalDataManager
from dataclasses import dataclass

from ..basic_types import SessionParams
from ..query_interface import InteractiveQuery

@dataclass
class LoopState:
    curr_str: str = None
    tvec: np.ndarray = None
    vec_state = None # 'VecState' class
    # model: OnlineModel = None
    knn_model = None # 'SimpleKNNRanker'

class LoopBase:
    q: InteractiveQuery
    params: SessionParams
    state: LoopState

    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        self.gdm = gdm
        self.params = params
        self.state = LoopState()
        self.q = q
        self.index = self.q.index

    @staticmethod
    def from_params(gdm, q, params) -> 'LoopBase':
        pass

    def set_text_vec(self, tvec):
        raise NotImplementedError('implement me in subclass')

    def next_batch(self):
        raise NotImplementedError('implement me in subclass')

    def refine(self):
        raise NotImplementedError('implement me in sublcass')


