from ..logistic_regression import LogisticRegressionPT
from .point_based import *
from .loop_base import *

class RocchioUpdate(PointBased):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        super().__init__(gdm, q, params)
        self.model = None
        self.alpha = params.method_config['alpha']
        self.beta = params.method_config['beta']
        self.gamma = params.method_config['gamma']

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        return RocchioUpdate(gdm, q, params)

    def set_text_vec(self, vec):
        super().set_text_vec(vec)
        self.curr_vec = vec
        self.model = None

    # def set_text_vec(self) # let super do this
    def refine(self, change=None):
        Xt, yt = self.q.getXy()
        
        """
        ## page 182 IR book (Raghavan)
          q = \alpha  q_0 + \beta mean rel - \gamma mean non rel
        """

        relX = Xt[yt > 0]
        sum_dr = relX.sum(axis=0)

        nrelX = Xt[yt == 0]
        sum_ndr = nrelX.sum(axis=0)

        mean_dr = sum_dr / (relX.shape[0] if relX.shape[0] > 0 else 1. )
        mean_ndr = sum_ndr / (nrelX.shape[0] if nrelX.shape[0] > 0 else 1.)

        self.curr_vec = self.alpha * self.curr_qvec  + self.beta * mean_dr - self.gamma * mean_ndr