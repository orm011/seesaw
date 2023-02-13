from .loop_base import *

class PointBased(LoopBase):
    def __init__(self, gdm, q, params):
        super().__init__(gdm, q, params)

    def set_text_vec(self, vec):
        super().set_text_vec(vec)
        self.state.tvec = vec
        self.curr_vec = vec

    def refine(self, change=None):
        raise NotImplementedError('implement in subclass')

    def next_batch(self):
        return self._next_batch_curr_vec(self.curr_vec)

class Plain(PointBased):
    def __init__(self, gdm, q, params):
        super().__init__(gdm, q, params)

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        return Plain(gdm, q, params)

    def refine(self, change=None):
        pass # no feedback