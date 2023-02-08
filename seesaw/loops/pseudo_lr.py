from .log_reg import LogisticRegressionPT
from .point_based import PointBased
from .loop_base import *
from .util import makeXy

import numpy as np
from .graph_based import get_label_prop, KnnProp2


class PseudoLR(PointBased):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        super().__init__(gdm, q, params)
        self.options = self.params.interactive_options
        self.label_prop_params = self.options['label_prop_params']
        self.log_reg_params = self.options['log_reg_params']
        self.switch_over = self.options['switch_over']
        self.real_sample_weight = self.options['real_sample_weight']
        assert self.real_sample_weight >= 1.

        label_prop = get_label_prop(q, label_prop_params=self.label_prop_params)
        self.knn_based = KnnProp2(gdm, q, params, knn_model = label_prop)

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        return PseudoLR(gdm, q, params)

    def set_text_vec(self, tvec):
        super().set_text_vec(tvec)
        self.knn_based.set_text_vec(tvec)

    def refine(self, change=None):
        self.knn_based.refine() # label prop,. # if only positives wont do anything.
        # if negatives it will try to help (not clear it works that way)
        
        X, y, is_real = makeXy(self.index, self.knn_based.state.knn_model, sample_size=self.options['sample_size'])
        model = LogisticRegressionPT(regularizer_vector=self.state.tvec,  **self.log_reg_params)

        weights = np.ones_like(y)
        weights[is_real > 0] = self.real_sample_weight

        model.fit(X, y.reshape(-1,1), weights.reshape(-1,1)) # y 
        self.curr_vec = model.get_coeff().reshape(-1)

    def next_batch(self):
        pos, neg = self.q.getXy(get_positions=True)

        if self.switch_over:
            if (len(pos) == 0 or len(neg) == 0):
                print('not switching over yet')
                return self.knn_based.next_batch() # tree based 
            else:
                return super().next_batch() # point based result
        else:
            return super().next_batch()

