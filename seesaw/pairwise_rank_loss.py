import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim

import numpy as np
from .search_loop_models import compute_inversions

class RankAndLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, inputs, labels, margin):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        npos = (labels == 1.).sum()
        nneg = (labels != 1.).sum()
        assert npos + nneg == labels.shape[0]
        if npos == 0:
            ctx.save_for_backward(weight, inputs, None, labels, margin)
            return torch.tensor(0.)
        elif nneg ==0:
            ctx.save_for_backward(weight, inputs, None, labels, margin)
            return torch.tensor(0.)
                
        scores = inputs @ weight    
        pos_scores = scores[labels == 1.].reshape(-1,1)
        neg_scores = scores[labels == 0.].reshape(1,-1)
        #print('p: ', pos_scores)
        #print('n: ', neg_scores)
        
        pos_scores = pos_scores - margin
        differences = -(pos_scores - neg_scores)
        losses = differences.clamp(min=0)
                
        #print('l: ', losses)
        loss = losses.reshape(-1).mean()
        ctx.save_for_backward(weight, inputs, scores, labels, margin)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        weight, inputs, scores, labels, margin = ctx.saved_tensors

        npos = (labels == 1.).sum()
        nneg = (labels != 1.).sum()
        #grad_input = grad_output.clone()
        if npos == 0 or nneg == 0:
            return torch.zeros_like(weight), None, None, None
        
        k = (1./npos) * (1./nneg)
        
        #print(scores)
        scores[labels == 1] -= margin # for inversions
        #print(scores)
        inversions = torch.from_numpy(compute_inversions(labels.numpy(), scores.numpy()))
        #print(inversions)

        ## use (possibly) sparse set of vectors with inversions
        non_zero_coeffs = inversions > 0
        inversions = inversions[non_zero_coeffs]
        labels = labels[non_zero_coeffs]
        inputs = inputs[non_zero_coeffs]

        signs = labels*2 - 1.
        coeffs = k * inversions * signs
        delta = -(coeffs.reshape(-1,1)*inputs).sum(axis=0)
        assert delta.shape == weight.shape
        return delta.reshape(-1)*grad_output, None, None, None
    
    
class RankLoss(nn.Module):
    def __init__(self, w, margin: float):
        super().__init__()
        self.w = nn.Parameter(data=F.normalize(w, dim=-1))
        self.margin = margin
        
    def forward(self, dat, labels):
        # dat = F.normalize(dat,dim=-1)
        # with torch.no_grad():
        #     print(self.w.norm())
        #w = F.normalize(self.w, dim=-1)
        ## normalizing seems to affect gradients quite a bit even when 
        ## the norm is 1. so, don't normalize within gradient.
        loss = RankAndLoss.apply(self.w, dat, labels, torch.tensor(self.margin))
        return loss
    
class VecState:
    def __init__(self, w : np.ndarray, margin : float, opt_class, opt_params={}, renormalize=False):
        w = torch.from_numpy(w).type(torch.float32).reshape(-1)
        self.mod = RankLoss(w, margin)
        self.opt = opt_class([self.mod.w], **opt_params)
        self.renormalize=renormalize
        
    def get_vec(self):
        return self.mod.w.detach().numpy()
    
    def update(self, vecs, labels):
        opt = self.opt
        mod = self.mod
        opt.zero_grad()
        v = torch.from_numpy(vecs).float()
        lab = (torch.from_numpy(labels) == 1).float()
        loss = mod(v, lab)
        loss.backward()
        opt.step()
        
        if self.renormalize:
            with torch.no_grad():
                self.mod.w.data = F.normalize(self.mod.w.data, dim=-1)



######### tests
from .search_loop_models import make_hard_neg_ds, LookupVec, DataLoader

def method0(w, dat, labels, margin):
    w = F.normalize(w, dim=-1)

    w = w.clone().detach().requires_grad_(True)
    w.retain_grad()
    w1 = w #F.normalize(w,dim=-1)
    loss = RankAndLoss.apply(w1, dat, labels, torch.tensor(margin))
    loss.backward()
    return loss.clone().detach(), w.grad.data.clone()

def method1(w, dat, labels, margin):
    w = F.normalize(w, dim=-1)
    npos = (labels == 1.).sum()
    nneg = labels.shape[0] - npos
    max_size = npos*nneg
    ds = make_hard_neg_ds(dat, labels, max_size=max_size,curr_vec=w)
    
    mod = LookupVec(dat.shape[1], margin=margin, optimizer=torch.optim.SGD, learning_rate=.01, init_vec=w)

    assert len(ds) == npos*nneg
    dl = DataLoader(ds, batch_size=len(ds))
    for b in dl:
        x = mod._batch_step(b, None)

    
    loss = x['loss']
    loss.backward()
    return loss.clone().detach(), mod.vec.grad.clone().reshape(-1)

def method2(w, data, labels,margin):
    w = F.normalize(w, dim=-1)
    mod = RankLoss(w, margin)
    opt = torch.optim.SGD([mod.w], lr=.01)
    opt.zero_grad()
    loss = mod(data, labels)
    loss.backward()
    return loss.clone().detach(), mod.w.grad.data.clone()


### test the three methods are equivalent
def random_test_equiv(dim, batch_size):
    w = torch.from_numpy(np.random.randn(dim)).float()
    dat = torch.from_numpy(np.random.randn(batch_size,dim)).float()
    w = F.normalize(w, dim=-1)
    dat = F.normalize(dat, dim=-1)
    
    nnegs = list(range(1,min(batch_size-1,5))) + [batch_size-1]
    
    for margin in [0., .05, .1, .2]:
        for nneg in nnegs:
            labels = torch.ones(dat.shape[0])
            labels[:nneg] = 0

            l0, g0 = method0(w.clone(), dat.clone(), labels,margin)
            l1, g1 = method1(w.clone(), dat.clone(), labels,margin)
            l2, g2 = method2(w.clone(), dat.clone(), labels,margin)

            assert torch.isclose(l0, l2), f'loss: {l0} vs {l2}'
            assert torch.isclose(l1, l2), f'loss: {l1} vs {l2}'
            assert torch.isclose(g0, g2, atol=1e-6).all(), f'gradient: {g0} vs {g2}'
            assert torch.isclose(g1, g2, atol=1e-6).all(),f'gradient: {g1} vs {g2}'

#random_test_equiv(dim=100, batch_size=200)