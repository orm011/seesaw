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

import math
class RegModule(nn.Module):
    def __init__(self, *, dim, xlx_matrix, qvec,
         label_loss_type,  
         reg_data_lambda, reg_norm_lambda, reg_query_lambda,
         use_qvec_norm,
         rank_loss_margin=0.,
         verbose=False, max_iter=100, lr=1.):
        super().__init__()
        layer = nn.Linear(dim, 1, bias=False)  # use it for initialization           
        self.weight = nn.Parameter(layer.weight.reshape(-1), requires_grad=True)

        assert label_loss_type in ['ce_loss', 'pairwise_rank_loss', 'pairwise_logistic_loss']
        self.label_loss_type = label_loss_type
        self.xlx_matrix = xlx_matrix
        self.qvec = F.normalize(qvec.reshape(-1), dim=-1)
        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose

        self.reg_query_lambda = reg_query_lambda
        self.use_qvec_norm = use_qvec_norm
        self.reg_norm_lambda = reg_norm_lambda
        self.reg_data_lambda = reg_data_lambda

        self.pos_weight = torch.tensor([1.])
        self.rank_loss_margin = rank_loss_margin

    def get_coeff(self):
        return F.normalize(self.weight.detach(), dim=-1).numpy()

    def forward(self, X, y=None):
        return X @ self.weight
    
    def _step(self, batch):
        if len(batch) == 2:
            X,y=batch # note y can be a floating point
            weight=None
        else:
            assert len(batch) == 3
            X,y,weight = batch
            
        logits = self(X, y)
        loss_labels= None
        if self.label_loss_type == 'ce_loss':
            weighted_celoss = F.binary_cross_entropy_with_logits(logits, y, weight=weight,
                                        reduction='none', pos_weight=self.pos_weight)        
            item_losses = weighted_celoss
        elif self.label_loss_type == 'pairwise_rank_loss':
            per_item_loss, max_inv = ref_pairwise_rank_loss(y, scores=logits, 
                                aggregate='sum',margin=self.rank_loss_margin, return_max_inversions=True)
            
            per_item_normalized = per_item_loss/max_inv
            item_losses = per_item_normalized
        elif self.label_loss_type == 'pairwise_logistic_loss':
            per_item_loss, max_inv = ref_pairwise_logistic_loss(y, scores=logits, 
                                aggregate='sum', return_max_inversions=True)
            
            per_item_normalized = per_item_loss/max_inv
            item_losses = per_item_normalized
        else:
            assert False


        # if self.use_qvec_norm:
        #     normvec = (self.weight - self.qvec)
        # else:
        #     normvec = self.weight

        nweight = F.normalize(self.weight, dim=-1)

        if self.use_qvec_norm:
            loss_norm = (self.weight.norm() - 1)**2
            loss_queryreg = (1 - nweight@self.qvec)/2.
        else:
            loss_norm = self.weight.norm()**2
            loss_queryreg = loss_norm*0.

        loss_datareg = nweight @ (self.xlx_matrix @ nweight)
        loss_labels = item_losses.sum()
        total_loss =  loss_labels + self.reg_data_lambda*loss_datareg + self.reg_norm_lambda*loss_norm + self.reg_query_lambda*loss_queryreg

        return {
            'loss_norm' :  self.reg_norm_lambda*loss_norm,
            'loss_labels': loss_labels,
            'loss_datareg': self.reg_data_lambda*loss_datareg,
            'loss_queryreg': self.reg_query_lambda*loss_queryreg,
            'loss': total_loss,
        }
    
    def training_step(self, batch, batch_idx):
        losses = self._step(batch)       
        return losses
    
    def validation_step(self, batch, batch_idx):
        losses = self._step(batch)
        return losses

    def configure_optimizers(self):
        return opt.LBFGS(self.parameters(), max_iter=self.max_iter, lr=self.lr, line_search_fn='strong_wolfe')

    def fit(self, X, y):
        trainer_ = BasicTrainer(mod=self, max_epochs=1)
        Xmu = X.mean(axis=0)
        X = X - Xmu.reshape(1,-1) # center this
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=X.shape[0], shuffle=True)

        losses_ = trainer_.fit(dl)
        if self.verbose:
            for ret in losses_:
                print(ret)
        return losses_
    

class MultiReg(PointBased):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        super().__init__(gdm, q, params)
        self.options = self.params.interactive_options
        xlx = get_weight_matrix_from_index(q.index, self.options['matrix_options'], xlx_matrix=True)
        self.xlx_matrix = torch.from_numpy(xlx).float()

    @staticmethod
    def from_params(gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        return MultiReg(gdm, q, params)

    def set_text_vec(self, tvec):
        super().set_text_vec(tvec)

    def refine(self, change=None):  
        X,y = self.q.getXy()      
        assert y.sum() > 0
        assert (y.shape[0] - y.sum()) > 0
        assert self.state.tvec is not None

        model_ = RegModule(dim=X.shape[1], xlx_matrix=self.xlx_matrix, qvec=torch.from_numpy(self.state.tvec).float(), 
                            label_loss_type=self.options['label_loss_type'], 
                            rank_loss_margin=self.options['rank_loss_margin'], 
                            reg_data_lambda=self.options['reg_data_lambda'], 
                            reg_norm_lambda=self.options['reg_norm_lambda'],
                            use_qvec_norm=self.options['use_qvec_norm'],
                            reg_query_lambda=self.options['reg_query_lambda'], 
                            verbose=self.options['verbose'], 
                            max_iter=self.options['max_iter'], 
                            lr=self.options['lr'])

        model_.fit(X,y)
        self.curr_vec = model_.get_coeff()

    def next_batch(self):
        return super().next_batch()
