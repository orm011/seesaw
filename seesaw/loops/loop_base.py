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
        self.curr_qvec = None
        self.reversal = False # will be modified by session
        self.started = False
        if self.params.start_policy == 'from_start':
            self.started = True

    def set_reversals(self):
        if not self.reversal:
            print('first reversal seen...')
            self.reversal = True

    def get_stats(self):
        return None

    def set_text_vec(self, vec):
        self.curr_qvec = vec

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
        if self.started:
            print('start met. next batch from custom method...')
            return self.next_batch()
        else:
            print('start not yet met. next batch using default...')
            return self._next_batch_curr_vec(vec=self.curr_qvec)

    def next_batch(self):
        ''' meant to be called  only here'''
        raise NotImplementedError('implement me in subclass')

    def refine(self, change=None):
        raise NotImplementedError('implement me in subclass')

    def refine_external(self, change=None):
        matchdf = self.q.getXy()
        X = self.q.index.vectors[matchdf.index.values]
        y = matchdf.ys.values
        by_image = matchdf.groupby('dbidx').ys.max()

        len_pos = (by_image == 1.).sum()
        len_neg = (by_image == 0.).sum()

        if self.params.start_policy == 'from_start':
            assert self.started
            start_condition = True
        elif self.params.start_policy == 'after_first_batch':
            start_condition = (len_pos + len_neg ) > 0
        elif self.params.start_policy == 'after_first_positive':
            start_condition = len_pos > 0
        elif self.params.start_policy == 'after_first_negative':
            start_condition = len_neg > 0
        elif self.params.start_policy == 'after_first_positive_and_negative':
            start_condition = (len_pos > 0) and (len_neg  > 0)        
        elif self.params.start_policy == 'after_first_reversal':
            start_condition = self.reversal
        else:
            assert False, 'policy not implemented'

        self.started = self.started or start_condition
        if self.started:
            print('start condition met... refinining custom method...')
            self.refine(change=change)


