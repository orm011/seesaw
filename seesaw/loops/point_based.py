from .loop_base import *

class PointBased(LoopBase):
    def __init__(self, gdm, q, params):
        super().__init__(gdm, q, params)
        self.curr_vec = None

    def set_text_vec(self, vec):
        self.state.tvec = vec
        self.curr_vec = vec

    def refine(self, change=None):
        raise NotImplementedError('implement in subclass')

    def next_batch(self):
        s = self.state
        p = self.params

        vec = self.curr_vec
        rescore_m = lambda vecs: vecs @ vec.reshape(-1, 1)

        b = self.q.query_stateful(
            vector=vec,
            batch_size=p.batch_size,
            shortlist_size=p.shortlist_size,
            agg_method=p.agg_method,
            aug_larger=p.aug_larger,
            rescore_method=rescore_m,
        )

        return b

class Plain(PointBased):
    def __init__(self, gdm, q, params):
        super().__init__(gdm, q, params)

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        return Plain(gdm, q, params)

    def refine(self, change=None):
        pass # no feedback