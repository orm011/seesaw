from seesaw.knn_graph import get_weight_matrix
from .log_reg import LogisticRegressionPT
from .point_based import PointBased
from .loop_base import *
from .util import makeXy

import numpy as np
from .graph_based import get_label_prop, KnnProp2, get_weight_matrix_from_index, lookup_weight_matrix

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from seesaw.basic_trainer import BasicTrainer
import torch
from seesaw.basic_trainer import BasicTrainer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

import torch.nn as nn
from ..rank_loss import ref_pairwise_logistic_loss, ref_pairwise_rank_loss
import pandas as pd
import math

from .multi_reg_module import MultiRegModule

class MultiRegNeg(PointBased):
    ### need qstr for the two classes.
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        super().__init__(gdm, q, params)
        self.options = self.params.interactive_options
        xlx = get_weight_matrix_from_index(q.index, self.options['matrix_options'], xlx_matrix=True)
        self.xlx_matrix = torch.from_numpy(xlx).float()
        self.confusion_vec = None

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        return MultiRegNeg(gdm, q, params)

    def set_text_vec(self, tvec):
        super().set_text_vec(tvec)
        ## run optimization based on regularization losses
        if self.options['reg_data_lambda'] > 0 and self.options['reg_query_lambda'] > 0 and self.started: # not well defined otherwise
            self.refine()
        else:
            self.curr_vec = self.curr_qvec


    def refine(self, change=None):  
        ## should it come from the labeldb itself
        ##other_names = {x for x in all_descs if x != self.}

        matchdf = self.q.getXy(target_description=None)
        X = self.q.index.vectors[matchdf.index.values]
        y = matchdf.ys.values


        box_df = self.q.label_db.get_box_df(return_description=True)
        descs = box_df[box_df.marked_accepted == 0].description.unique()
#        if len(descs) > 0:
#            print('refining conf vec..')
        if len(descs) > 0:
            alt_desc = descs[0]
            print(f'{alt_desc=}')
            #curr_confvec = self.index.string2vec(alt_desc)
            conf_df = self.q.getXy(target_description=alt_desc)
            # Xconf = self.q.index.vectors[conf_df.index.values]
            yconf = conf_df.ys.values
        else:
            yconf = np.zeros_like(y)
        ys = np.stack([y, yconf], axis=1).astype('float32')
        
        assert ys.shape[0] == y.shape[0]
        assert ys.shape[1] == 2

        print(f'{ys.sum(axis=0)=}')
        assert self.curr_qvec is not None


        model_ = MultiRegModule(qvec=torch.from_numpy(self.curr_qvec).float(), 
                            reg_norm_lambda=self.options['reg_norm_lambda'],
                            reg_query_lambda=self.options['reg_query_lambda'], 
                            verbose=self.options['verbose'], 
                            max_iter=self.options['max_iter'], 
                            lr=self.options['lr'])

        model_.fit(X, ys, matchdf)
        self.curr_vec = model_.get_coeff()
        self.confusion_vec = F.normalize(model_.weight[1].detach(), dim=-1).numpy()

    def next_batch(self): 
        
        ### try1 : just remove scores from the other class.        
        ## rescore method is within query stateful
        # def rescore_fn(vecs):
        #     score1 = vecs @ self.curr_vec.reshape(-1,1)
        #     score2 = vecs @ self.confusion_vec.reshape(-1,1)
        #     return score1 - score2

        b = self.q.query_stateful(
            vector=self.curr_vec,
            batch_size=self.params.batch_size,
            shortlist_size=self.params.shortlist_size,
            agg_method=self.params.agg_method,
            aug_larger=self.params.aug_larger,
            rescore_method=None,
            vector2 = self.confusion_vec if self.options['discount_neg'] else np.zeros(512)
        )

        return b 
