from ..logistic_regression import LogisticRegressionPT
from .point_based import *
from .loop_base import *

class LogReg2(PointBased):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        super().__init__(gdm, q, params)
        self.model = None

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        return LogReg2(gdm, q, params)

    def set_text_vec(self, vec):
        super().set_text_vec(vec)
        self.curr_vec = vec
        self.model = None

    # def set_text_vec(self) # let super do this
    def refine(self, change=None):
        Xt, yt = self.q.getXy()
        
        if self.model is None:
            self.model = LogisticRegressionPT(regularizer_vector=self.state.tvec, **self.params.interactive_options)

        ## if there are only positives, fitting should already do nothing due to regularization... except loss is not the same.
        if (yt == 1).all():
            print('doing nothing, only positives')
        elif (yt == 0).all():
            print('doing nothing, only negatives')
        else:
            self.model.fit(Xt, yt.reshape(-1,1))
            self.curr_vec = self.model.get_coeff()
