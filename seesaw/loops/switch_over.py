from .loop_base import *

class SwitchOver(LoopBase):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams, method0: LoopBase, method1: LoopBase):
        super().__init__(gdm, q, params)
        self.method0 = method0
        self.method1 = method1

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        assert params.interactive == 'switch_over'
        params0 = params.copy()
        params1 = params.copy()

        opts = params.interactive_options
        opts0 = opts['method0']
        opts1 = opts['method1']

        params0.interactive = opts0['interactive']
        params0.interactive_options = opts0['interactive_options']

        params1.interactive=opts1['interactive']
        params1.interactive_options=opts1['interactive_options']

        return SwitchOver(gdm, q, params, 
                method0=LoopBase.from_params(gdm, q, params0), 
                method1=LoopBase.from_params(gdm, q, params1)
            )

    def switch_condition(self):
        pos, neg = self.q.getXy(get_positions=True)
        return (len(pos) > 0 and len(neg) > 0)

    def set_text_vec(self, tvec):
        self.method0.set_text_vec(tvec)
        self.method1.set_text_vec(tvec)

    def refine(self):
        self.method0.refine()
        self.method1.refine()

    def next_batch(self):
        if self.switch_condition(): # use method 1 once condition met
            return self.method1.next_batch_external()
        else:
            return self.method0.next_batch_external()

