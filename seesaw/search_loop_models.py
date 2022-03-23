import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Subset
from tqdm.auto import tqdm
import pyroaring as pr
import sys
from torch.utils.data import TensorDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import os
import pytorch_lightning as pl
from .pairwise_rank_loss import compute_inversions


class CustomInterrupt(pl.callbacks.Callback):
    def on_keyboard_interrupt(self, trainer, pl_module):
        raise InterruptedError('custom')

class CustomTqdm(pl.callbacks.progress.ProgressBar):
    def init_train_tqdm(self):
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            miniters=40,
        )
        return bar

class PTLogisiticRegression(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        bias: bool = True,
        learning_rate: float = 5e-3,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        C: float = 0.0,
        positive_weight : float = 1.0,
        **kwargs
    ):
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='Adam')
            l1_strength: L1 regularization strength (default=None)
            l2_strength: L2 regularization strength (default=None)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.linear = nn.Linear(in_features=self.hparams.input_dim, 
                                out_features=1,
                                bias=bias)
        
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hparams.positive_weight]).float(), reduction='none')
        self.average_precision = pl.metrics.AveragePrecision(num_classes=2, pos_label=1, 
                                                               compute_on_step=False)

    def forward(self, qvec):
        qvec = self.linear(qvec)
        return qvec
    
    def _batch_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.loss(logits, y.view(-1,1)).reshape(-1).mean()
        
        # L2 regularizer
        if self.hparams.C > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss = self.hparams.C*loss + l2_reg

        loss /= y.size(0)
        return {'loss':loss, 'logits':logits, 'y':y}

    def training_step(self, batch, batch_idx):
        d = self._batch_step(batch, batch_idx)
        loss = d['loss']
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        d = self._batch_step(batch, batch_idx)
        loss = d['loss']
        logits = torch.clone(d['logits']).view(-1)
        
        self.average_precision(logits, d['y'].view(-1).long())
        self.log('loss/val', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'logits':logits, 'y':d['y']}
    
    def validation_epoch_end(self, validation_step_outputs):
        apval = self.average_precision.compute()
        self.log('AP/val', apval, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.average_precision.reset()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)


def fit_reg(*, mod, X, y, batch_size, valX=None, valy=None, logger=None,  max_epochs=4, gpus=0, precision=32):    
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    
    train_ds = TensorDataset(X,torch.from_numpy(y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    if valX is not None:
        if not torch.is_tensor(valX):
            valX = torch.from_numpy(valX)
        val_ds = TensorDataset(valX, torch.from_numpy(valy))
        es = [pl.callbacks.early_stopping.EarlyStopping(monitor='AP/val', mode='max', patience=3)]
        val_loader = DataLoader(val_ds, batch_size=2000, shuffle=False, num_workers=0)
    else:
        val_loader = None
        es = []

    trainer = pl.Trainer(logger=None, 
                         gpus=gpus, precision=precision, max_epochs=max_epochs,
                         callbacks =[CustomInterrupt()], 
                         checkpoint_callback=False,
                         weights_save_path='/tmp/',
                         progress_bar_refresh_rate=0, #=10
                        )
    trainer.fit(mod, train_loader, val_loader)

class LookupVec(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        margin: float = .1,
        init_vec: torch.tensor = None,
        learning_rate: float = 5e-3,
        positive_weight: float = 1.,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        **kwargs
    ):
        """
        Args:
        input_dim: number of dimensions of the input (at least 1)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        
        if init_vec is not None:
            self.vec = nn.Parameter(init_vec.reshape(1,-1))
        else:
            t = torch.randn(1,input_dim)
            self.vec = nn.Parameter(t/t.norm())


        # self.loss = nn.CosineEmbeddingLoss(margin,reduction='none')
        self.rank_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, qvec):
        vec = F.normalize(self.vec.reshape(-1), dim=-1)
        return qvec @ vec.reshape(-1) # qvecs are already normalized
        # return F.cosine_similarity(self.vec, qvec)

    def _batch_step(self, batch, batch_idx):
        X1, X2, y = batch
        sim1 = self(X1)
        sim2 = self(X2)
        
        losses = self.rank_loss(sim1, sim2, y.view(-1)).reshape(-1)
        return {'loss':losses.mean(),  'y':y}

    def training_step(self, batch, batch_idx):
        d = self._batch_step(batch, batch_idx)
        loss = d['loss']
        # self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        d = self._batch_step(batch, batch_idx)
        loss = d['loss']
        
        # self.log('loss/val', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'y':d['y']}
    
    def validation_epoch_end(self, validation_step_outputs):
        apval = self.average_precision.compute()
        self.log('AP/val', apval, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.average_precision.reset()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

def make_hard_neg_ds(X,y, max_size, curr_vec):
    neg_flag = y < 1.
    pos_flag = y > 0.
    positions = torch.arange(y.shape[0])
    scores = X @ curr_vec.reshape(-1,1)

    neg_idx = positions[neg_flag]
    neg_scores = scores[neg_flag]
    hard_neg = torch.argsort(neg_scores.reshape(-1), descending=True)[:max_size]
    hard_neg_idx = neg_idx[hard_neg]

    pos_idx = positions[pos_flag]# torch.arange(y.shape[0])[pos_flag]
    pos_scores = scores[pos_flag]
    hard_pos = torch.argsort(pos_scores.reshape(-1), descending=False)[:max_size]
    hard_pos_idx = pos_idx[hard_pos]

    assert (y[hard_pos_idx] > 0.).all()
    assert (y[hard_neg_idx] < 1.).all()

    if hard_neg_idx.shape[0] >= 2:
        assert scores[hard_neg_idx[0]] >= scores[hard_neg_idx[1]]

    if hard_pos_idx.shape[0] >= 2:
        assert scores[hard_pos_idx[0]] <= scores[hard_pos_idx[0]]
    
    root = int(math.ceil(math.sqrt(max_size)))

    if hard_pos_idx.shape[0] * hard_neg_idx.shape[0] > max_size:
        pos_bound = root
        neg_bound = root
        if hard_pos_idx.shape[0] < root:
            neg_bound = int(math.ceil(max_size/hard_pos_idx.shape[0]))

        if hard_neg_idx.shape[0] < root:
            pos_bound = int(math.ceil(max_size/hard_neg_idx.shape[0]))
                            
        hard_pos_idx = hard_pos_idx[:pos_bound]
        hard_neg_idx = hard_neg_idx[:neg_bound]
        assert hard_pos_idx.shape[0]*hard_neg_idx.shape[0] >= max_size
        assert (hard_pos_idx.shape[0]-1)*(hard_neg_idx.shape[0]-1) < max_size

    X1ls = []
    X2ls = []
    for pi in hard_pos_idx:
        for nj in hard_neg_idx: 
            X1ls.append(X[pi])
            X2ls.append(X[nj])
    #print(hard_pos_idx.shape, hard_neg_idx.shape)


    X1 = torch.stack(X1ls)
    X2 = torch.stack(X2ls)
    train_ds = TensorDataset(X1,X2, torch.ones(X1.shape[0]))
    
    return train_ds

def make_tuple_ds(X, y, max_size):
    X1ls = []
    X2ls = []
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if y[i] > y[j]:
                X1ls.append(X[i])
                X2ls.append(X[j])

    X1 = torch.stack(X1ls)
    X2 = torch.stack(X2ls)
    train_ds = TensorDataset(X1,X2, torch.ones(X1.shape[0]))
    if len(train_ds) > max_size:
        ## random sample... # should prefer some more
        randsel = torch.randperm(len(train_ds))[:max_size]
        train_ds = Subset(train_ds, randsel)
    return train_ds


def fit_rank2(*, mod, X, y, batch_size, max_examples, valX=None, valy=None, logger=None, margin=.0, max_epochs=4, gpus=0, precision=32):

    ## for running on spc. 
    if os.environ.get("SLURM_NTASKS", '') != '':
        del os.environ["SLURM_NTASKS"]
    if os.environ.get("SLURM_JOB_NAME", '') != '':
        del os.environ["SLURM_JOB_NAME"]

    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    
    assert (y >= 0).all()
    assert (y <= 1).all()

    #train_ds = make_hard_neg_ds(X, y, max_size=max_examples, curr_vec=mod.vec.detach())
    # ridx,iis,jjs = hard_neg_tuples(mod.vec.detach().numpy(), X.numpy(), y, max_tups=max_examples)
    # train_ds = TensorDataset(X[ridx][iis],X[ridx][jjs], torch.ones(X[ridx][iis].shape[0]))
    train_ds = hard_neg_tuples_faster(mod.vec.detach().numpy(), X.numpy(), y, max_tups=max_examples, margin=margin)
    ## want a tensor with pos, neg, 1. ideally the highest scored negatives and lowest scored positives.

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    if valX is not None:
        if not torch.is_tensor(valX):
            valX = torch.from_numpy(valX)
        val_ds = TensorDataset(valX, torch.from_numpy(valy))
        es = [pl.callbacks.early_stopping.EarlyStopping(monitor='AP/val', mode='max', patience=3)]
        val_loader = DataLoader(val_ds, batch_size=2000, shuffle=False, num_workers=0)
    else:
        val_loader = None
        es = []


    trainer = pl.Trainer(logger=None,
                         gpus=gpus, precision=precision, max_epochs=max_epochs,
                         #callbacks =[],
                         callbacks= [ CustomInterrupt()],
                           # CustomTqdm()#  ], 
                         checkpoint_callback=False,
                         weights_save_path='/tmp/',
                         progress_bar_refresh_rate=0, #=10
                        )
    trainer.fit(mod, train_loader, val_loader)


def adjust_vec(vec, Xt, yt, learning_rate, loss_margin, max_examples, minibatch_size):
    ## cosine sim in produce
    vec = torch.from_numpy(vec).type(torch.float32)
    mod = LookupVec(Xt.shape[1], margin=loss_margin, optimizer=torch.optim.SGD, learning_rate=learning_rate, init_vec=vec)
    fit_rank2(mod=mod, X=Xt.astype('float32'), y=yt.astype('float'), 
            max_examples=max_examples, batch_size=minibatch_size,max_epochs=1, margin=loss_margin)
    newvec = mod.vec.detach().numpy().reshape(1,-1)
    return newvec 




def max_inversions_given_max_tups(labs, inversions, max_tups):
    orig_df = pd.DataFrame({'labs':labs, 'inversions':inversions})
    ddf = orig_df.sort_values('inversions', ascending=False)

    pdf = ddf[ddf.labs]
    ndf = ddf[~ddf.labs]

    ncutoff = pdf.shape[0]
    pcutoff = ndf.shape[0]

    def total_inversions(ncutoff,pcutoff):
        if ncutoff <= pcutoff:
            tot = np.minimum(ndf.inversions.values[:ncutoff], pcutoff).sum()
        else:
            tot = np.minimum(pdf.inversions.values[:pcutoff], ncutoff).sum()

        return tot

    curr_inv = total_inversions(ncutoff,pcutoff)

    while (ncutoff*pcutoff > max_tups) and (ncutoff > 1 or pcutoff > 1):
        tot1 = total_inversions(max(1,ncutoff-1), pcutoff)
        tot2 = total_inversions(ncutoff, max(1,pcutoff-1))
        if tot2 >= tot1:
            pcutoff-=1
        else:
            ncutoff-=1

    ncutoff = max(1,ncutoff)
    pcutoff = max(1,pcutoff)

    tot_inv = total_inversions(ncutoff, pcutoff)
    tot_tup = ncutoff*pcutoff

    pidx = pdf.index.values[:pcutoff]
    nidx = ndf.index.values[:ncutoff]
    return pidx, nidx, tot_inv, tot_tup

def hard_neg_tuples_faster(v, Xt, yt, max_tups, margin):
    """returns indices for the 'hardest' ntups
    """
    ### compute reversals for each vector
    ### keep reversals about equal?
    labs = (yt == 1.) # make boolean
    scores = (Xt @ v.reshape(-1,1)).reshape(-1)
    scores[labs] -= margin 
    inversions = compute_inversions(labs, scores)
    pidx, nidx, _, _ = max_inversions_given_max_tups(labs, inversions, max_tups)
    pidx, nidx = np.meshgrid(pidx, nidx)
    pidx = pidx.reshape(-1)
    nidx = nidx.reshape(-1)
    assert labs[pidx].all()
    assert ~labs[nidx].any()
    
    dummy = torch.ones(size=(pidx.shape[0],1))
    return TensorDataset(torch.from_numpy(Xt[pidx]), torch.from_numpy(Xt[nidx]), dummy)


def hard_neg_tuples(v, Xt, yt, max_tups):
    """returns indices for the 'hardest' ntups
    """
    p = np.where(yt > 0)[0]
    n = np.where(yt < 1)[0]
    assert p.shape[0] > 0
    assert n.shape[0] > 0
    
    scores = Xt @ v.reshape(-1,1)
    score_diffs = (scores[p].reshape(-1,1) - scores[n].reshape(1,-1))
    iis,jjs = np.meshgrid(np.arange(p.shape[0]), np.arange(n.shape[0]), indexing='ij')
    diff_order = np.argsort(score_diffs, axis=None)[:max_tups]
    #   score_diffs.flatten()[diff_order]
    pps = p[iis.flatten()[diff_order]]
    nns = n[jjs.flatten()[diff_order]]

    ridx = np.array(pr.BitMap(pps).union(pr.BitMap(nns)))
    lookup_tab = np.zeros(Xt.shape[0], dtype='int') -1
    lookup_tab[ridx] = np.arange(ridx.shape[0], dtype='int')
    piis = lookup_tab[pps]
    pjjs = lookup_tab[nns]
    # then X[ridx][piis] and X[ridx][jjs]
    # rdix o piis == iis <=> piis = iis
    assert (ridx[piis] == pps).all()
    return ridx, piis, pjjs

#import cvxpy as cp
def adjust_vec2(v, Xt, yt, *, max_examples, loss_margin=.1, C=.1, solver='SCS'):
    import cvxpy as cp
    # y = evbin.query_ground_truth[cat][dbidxs].values
    # X = hdb.embedded[dbidxs]
    margin = loss_margin
    nump = (yt > 0).sum()
    numn = (yt < 1).sum()

    ridx, piis,pjjs = hard_neg_tuples(v, Xt, yt, max_tups=max_examples)    
    # hnds  = make_hard_neg_ds(torch.from_numpy(Xt),torch.from_numpy(yt).float(),max_size=max_examples,curr_vec=v)    
    def rank_loss(s1, s2, margin):
        loss = cp.sum(cp.pos(- (s1 - s2 - margin)))
        return loss

    wstr = (v/np.linalg.norm(v)).reshape(-1)
    w = cp.Variable(shape=wstr.shape, name='w')
    w.value = wstr
    nweight = (nump + numn)/(nump*numn) # increase with n but also account for tuple blowup wrt n
    X = Xt[ridx] # reduce problem size here

    scores = X @ w
    obj = rank_loss(scores[piis], scores[pjjs], loss_margin)*nweight + C*(1. - (w@wstr))
    prob = cp.Problem(cp.Minimize(obj), constraints=[cp.norm2(w) <= 1.])
    prob.solve(warm_start=True, solver=solver)
    return w.value.copy().reshape(1,-1)