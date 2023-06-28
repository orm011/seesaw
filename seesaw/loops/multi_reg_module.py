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


def compound_loss(logits, ys):
    vertical = F.binary_cross_entropy_with_logits(logits, ys, reduction='none')
    total_vertical = vertical.mean()
    
    near_misses = ys.sum(dim=1)
    horizontal = F.cross_entropy(logits[near_misses > 0], ys[near_misses > 0], reduction='none')
    total_horizontal = horizontal.sum()
    
    total_near = near_misses.sum()
    if total_near == 0:
        total_near +=1
        
    return {'vertical':total_vertical, 'horizontal':total_horizontal/total_near}

class MultiRegModule(nn.Module):
    def __init__(self, *, qvec, qvec2=None,
        reg_norm_lambda, reg_query_lambda,
         verbose=False, max_iter=100, lr=1.):
        super().__init__()

        assert not math.isclose(qvec.norm().item() , 0.)
        self.qvec = F.normalize(qvec.reshape(-1), dim=-1)
        #self.qvec2 = F.normalize(qvec2.reshape(-1), dim=-1)

        self.linear = nn.Linear(in_features=512, out_features=2, bias=False)
        self.weight = self.linear.weight

        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose

        self.reg_query_lambda = reg_query_lambda
        self.reg_norm_lambda = reg_norm_lambda

    def get_coeff(self):
        return F.normalize(self.weight[0,:].detach(), dim=-1).numpy()

    def forward(self, X, y=None):
        return self.linear(X)
    
    def _step(self, batch):
        assert not self.linear.weight.isnan().any(), f'{self.weight=}'

        if len(batch) == 2:                
            X,y=batch # note y can be a floating point
            if y is None:
                sample_weight = None
            else:

                sample_weight = torch.ones_like(y)
        elif len(batch) == 3:
            X,y,sample_weight = batch
        else:
            assert False

        sample_weight = sample_weight.float()

        item_losses = self.weight.sum()*0
        assert y.shape[1] == 2 ## target, conf1, rest
        if X is not None:
            assert not y.isnan().any()
            assert not X.isnan().any()
            assert not sample_weight.isnan().any()

            norm_weights = F.normalize(self.weight, dim=1)
            logits = X @ norm_weights.t()

            vertical = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            total_vertical = vertical.sum(dim=1) 
            assert total_vertical.shape[0] == X.shape[0]

    
            near_misses = y.sum(dim=1)
            horizontal = F.cross_entropy(logits[near_misses > 0], y[near_misses > 0], reduction='none')


            total_near = (near_misses >0).sum()
            if total_near == 0:
                total_near +=1

            vertical_sum = total_vertical@sample_weight
            horizontal_sum = horizontal @ sample_weight[near_misses > 0]



        norm_weight = norm_weights[0]
        #loss_norm = (self.weight.norm() - 1)**2
        loss_labels = vertical_sum + horizontal_sum
        loss_norm = self.reg_norm_lambda *  ( torch.cosh( (self.weight.norm(dim=1)).log() ) - 1. ).sum()
        #loss_datareg = self.reg_data_lambda * ( norm_weight @ ( self.xlx_matrix @ norm_weight ) )
        loss_queryreg = self.reg_query_lambda * ( ( 1 - norm_weight@self.qvec )/2. )
        loss_queryreg2 = self.reg_query_lambda * ( ( 1 - norm_weights[1]@self.qvec )/2. ) 

        total_loss = loss_labels + loss_norm + loss_queryreg + loss_queryreg2

        ans =  {
            'loss_norm' : loss_norm,
#            'loss_datareg': loss_datareg,
            'loss_queryreg': loss_queryreg,
            'loss_queryreg2':loss_queryreg2,
            'vertical_loss': vertical_sum,
            'horizontal_loss': horizontal_sum,
            'loss': total_loss,
        }

        assert not total_loss.isnan(), f'{ans=}'
        return ans
    
    def training_step(self, batch, batch_idx):
        losses = self._step(batch)       
        return losses
    
    def validation_step(self, batch, batch_idx):
        losses = self._step(batch)
        return losses

    def configure_optimizers(self):
        return opt.LBFGS(self.parameters(), max_iter=self.max_iter, lr=self.lr, line_search_fn='strong_wolfe')

    def fit(self, X, y, matchdf):
        trainer_ = BasicTrainer(mod=self, max_epochs=1, verbose=self.verbose)
        
        ## 1/(nvecs for image)

        vecs_per_image = matchdf.groupby('dbidx').size().rename('num_vecs').reset_index()
        matchdf = pd.merge(matchdf, vecs_per_image, left_on='dbidx', right_on='dbidx')
        vec_weight = matchdf.num_vecs.astype('float').pow(-1).values

        if X.shape[0] > 0:
            Xmu = X.mean(axis=0)
            X = X - Xmu.reshape(1,-1) # center this
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(vec_weight))
            dl = DataLoader(ds, batch_size=X.shape[0], shuffle=True)
        else:
            dl= None

        losses_ = trainer_.fit(dl)
        if self.verbose:
            df = pd.DataFrame.from_records(losses_)
            agg_df= df.groupby('k').mean()
            print(agg_df)
        return losses_
    
