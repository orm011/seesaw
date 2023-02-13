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
        self.curr_vec = None

    def set_text_vec(self, vec):
        self.curr_vec = vec

    def _next_batch_curr_vec(self, vec):
        rescore_m = lambda vecs: vecs @ vec.reshape(-1, 1)

        b = self.q.query_stateful(
            vector=vec,
            batch_size=self.params.batch_size,
            shortlist_size=self.params.shortlist_size,
            agg_method=self.params.agg_method,
            aug_larger=self.params.aug_larger,
            rescore_method=rescore_m,
        )

        return b   

    @staticmethod
    def from_params(gdm, q, params) -> 'LoopBase':
        pass


    def next_batch_external(self):
        if self.params.start_policy == 'from_start':
            return self.next_batch()
        elif self.params.start_policy == 'after_first_positive':
            pos, _ = self.q.getXy(get_positions=True)
            if len(pos) > 0: ## for now, stick to clip until found one positive result
                return self.next_batch()
            else:
                return self._next_batch_curr_vec(vec=self.curr_vec)
        elif self.params.start_policy == 'first_reversal':
            assert False
        else:
            assert False


    def next_batch(self):
        ''' meant to be called  only here'''
        raise NotImplementedError('implement me in subclass')

    def refine(self, change=None):
        raise NotImplementedError('implement me in sublcass')


