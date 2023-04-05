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

class RegModule2(nn.Module):
    def __init__(self, *, dim, xlx_matrix, qvec,
         label_loss_type,  
         reg_data_lambda, reg_norm_lambda, reg_query_lambda,
         use_qvec_norm,
         rank_loss_margin=0.,
         pos_weight,
         verbose=False, max_iter=100, lr=1.):
        super().__init__()

        assert label_loss_type in ['ce_loss', 'pairwise_rank_loss', 'pairwise_logistic_loss']
        self.label_loss_type = label_loss_type
        self.xlx_matrix = xlx_matrix
        assert not math.isclose(qvec.norm().item() , 0.)

        self.qvec = F.normalize(qvec.reshape(-1), dim=-1)
        self.weight = nn.Parameter(self.qvec.clone(), requires_grad=True)
        assert not self.weight.isnan().any()

        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose

        self.reg_query_lambda = reg_query_lambda
        self.use_qvec_norm = use_qvec_norm
        self.reg_norm_lambda = reg_norm_lambda
        self.reg_data_lambda = reg_data_lambda

        self.pos_weight = pos_weight #'balanced' #torch.tensor([1.])
        self.rank_loss_margin = rank_loss_margin

    def get_coeff(self):
        return F.normalize(self.weight.detach(), dim=-1).numpy()

    def forward(self, X, y=None):
        return X @ self.weight
    
    def _step(self, batch):
        assert not self.weight.isnan().any(), f'{self.weight=}'

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
        if X is not None:
            assert not y.isnan().any()
            assert not X.isnan().any()
            assert not sample_weight.isnan().any()

            item_losses = self.weight.sum()*torch.zeros_like(sample_weight)

            logits = self(X, y)
            orig_sum = sample_weight.sum()
            assert orig_sum > 0
            pos_total = (y == 1).float()@sample_weight
            neg_total = orig_sum - pos_total
            if self.label_loss_type == 'ce_loss':
                if self.pos_weight == 'balanced':
                    positive_weight = (neg_total+1.)/(pos_total + 1.)
                elif type(self.pos_weight) is float:
                    positive_weight = torch.tensor([self.pos_weight])
                else:
                    assert False, 'unknown pos weight type'
                    
                celoss = F.binary_cross_entropy_with_logits(logits, y, weight=None,
                                            reduction='none', pos_weight=None)

                sample_weight[y == 1] *= positive_weight
                sample_weight *= orig_sum/sample_weight.sum()

                assert torch.isclose(sample_weight.sum(), orig_sum)
                item_losses = celoss
            elif self.label_loss_type == 'pairwise_rank_loss':
                if pos_total > 0 and neg_total > 0:
                    per_item_loss, max_inv = ref_pairwise_rank_loss(y, scores=logits, 
                                        aggregate='sum',margin=self.rank_loss_margin, return_max_inversions=True)
                    
                    per_item_normalized = per_item_loss/max_inv
                    item_losses = per_item_normalized
            elif self.label_loss_type == 'pairwise_logistic_loss':
                if pos_total > 0 and neg_total > 0:
                    per_item_loss, max_inv = ref_pairwise_logistic_loss(y, scores=logits, 
                                        aggregate='sum', return_max_inversions=True)
                    
                    per_item_normalized = per_item_loss/max_inv
                    item_losses = per_item_normalized 
            else:
                assert False

            item_losses *= sample_weight
        
        normalized_weight = F.normalize(self.weight, dim=-1)


        #loss_norm = (self.weight.norm() - 1)**2
        loss_norm = self.reg_norm_lambda *  ( torch.cosh( (self.weight @ self.weight).log() ) - 1. )
        loss_datareg = self.reg_data_lambda * ( self.weight @ ( self.xlx_matrix @ self.weight ) )
        loss_queryreg = self.reg_query_lambda * ( ( 1 - normalized_weight@self.qvec )/2. )
        loss_labels = item_losses.sum()

        total_loss = loss_labels + loss_datareg + loss_norm + loss_queryreg

        ans =  {
            'loss_norm' : loss_norm,
            'loss_datareg': loss_datareg,
            'loss_queryreg': loss_queryreg,
            'loss_labels': loss_labels,
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
    

class MultiRegNeg(PointBased):
    ### need qstr for the two classes.
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        super().__init__(gdm, q, params)
        self.options = self.params.interactive_options
        xlx = get_weight_matrix_from_index(q.index, self.options['matrix_options'], xlx_matrix=True)
        self.xlx_matrix = torch.from_numpy(xlx).float()
        self.confusion_vec = np.zeros(512)

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

        #if self.options['random_as_negative']:
        print(f'{y.sum()=}')
        assert self.curr_qvec is not None

        model_ = RegModule2(dim=X.shape[1], xlx_matrix=self.xlx_matrix, qvec=torch.from_numpy(self.curr_qvec).float(), 
                            label_loss_type=self.options['label_loss_type'], 
                            rank_loss_margin=self.options['rank_loss_margin'], 
                            reg_data_lambda=self.options['reg_data_lambda'], 
                            reg_norm_lambda=self.options['reg_norm_lambda'],
                            use_qvec_norm=self.options['use_qvec_norm'],
                            reg_query_lambda=self.options['reg_query_lambda'], 
                            verbose=self.options['verbose'], 
                            max_iter=self.options['max_iter'], 
                            pos_weight=self.options['pos_weight'],
                            lr=self.options['lr'])

        model_.fit(X, y, matchdf)
        self.curr_vec = model_.get_coeff()


        box_df = self.q.label_db.get_box_df(return_description=True)
        descs = box_df[box_df.marked_accepted == 0].description.unique()
        if len(descs) > 0 and self.options['discount_neg']:
            print('refining conf vec..')
            assert len(descs) == 1
            alt_desc = descs[0]
            print(f'{alt_desc=}')
            curr_confvec = self.index.string2vec(alt_desc)
            conf_df = self.q.getXy(target_description=alt_desc)
            Xconf = self.q.index.vectors[conf_df.index.values]
            yconf = conf_df.ys.values
            print(f'{yconf.sum()=}')    


            assert X.shape[0] == Xconf.shape[0]

            model_confusion = RegModule2(dim=X.shape[1], xlx_matrix=self.xlx_matrix, qvec=torch.from_numpy(curr_confvec).float(), 
                                label_loss_type=self.options['label_loss_type'], 
                                rank_loss_margin=self.options['rank_loss_margin'], 
                                reg_data_lambda=self.options['reg_data_lambda'], 
                                reg_norm_lambda=self.options['reg_norm_lambda'],
                                use_qvec_norm=self.options['use_qvec_norm'],
                                reg_query_lambda=self.options['reg_query_lambda'], 
                                verbose=self.options['verbose'], 
                                max_iter=self.options['max_iter'], 
                                pos_weight=self.options['pos_weight'],
                                lr=self.options['lr'])

            model_confusion.fit(Xconf, yconf, conf_df)
            self.confusion_vec = model_confusion.get_coeff()

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
            vector2 = self.confusion_vec
        )

        return b   
