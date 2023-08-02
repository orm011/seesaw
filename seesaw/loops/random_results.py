from .loop_base import LoopBase, GlobalDataManager, InteractiveQuery, SessionParams

class RandomResults(LoopBase):
    
    def set_text_vec(self, vec):
        self.curr_qvec = vec
        ## pick randomly from index. do not return seen 

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        return RandomResults(gdm, q, params)

    def next_batch_external(self):
        return self.q.query_random(batch_size=self.params.batch_size)

    def refine_external(self, change=None):
        pass

