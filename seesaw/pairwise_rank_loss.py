import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import numpy as np

def _positive_inversions(labs):
    return np.cumsum(~labs)*labs

def _negative_inversions(labs):
    labs = ~labs[::-1]
    return _positive_inversions(labs)[::-1]

def compute_inversions(labs, scores):
    assert labs.shape == scores.shape
    assert len(labs.shape) == 1
    labs = labs.astype('bool')
    scores = scores.copy()
    descending_order = np.argsort(-scores)
    labs = labs[descending_order]
    total_invs = _positive_inversions(labs) + _negative_inversions(labs)
    inv_order = np.argsort(descending_order)
    return total_invs[inv_order]


class RankAndLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, inputs, labels, margin, dummy_forward=False):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        labels = (labels == 1.)
        npos = labels.sum()
        nneg = (~labels).sum()
        assert npos + nneg == labels.shape[0]
        npairs = npos*nneg
        if npairs.item() == 0:
            loss = torch.tensor(0.)
            ctx.save_for_backward(weight, None, None, None, loss, npairs, None)
            return loss
                
        scores = inputs @ weight
        scores[labels] -= margin # remove margin from pos scores
        desc_order = torch.argsort(-scores)

        ordered_scores = scores[desc_order]
        ordered_labels = labels[desc_order]

        hardest_neg = (~ordered_labels).nonzero()[0].item()
        hardest_pos = ordered_labels.nonzero()[-1].item()

        if hardest_neg > hardest_pos:
            loss = torch.tensor(0.)
            ctx.save_for_backward(weight, None, None, None, loss, npairs, None)
            return loss

        relevant_indices = desc_order[hardest_neg:hardest_pos+1]

        scores = scores[relevant_indices]
        labels = labels[relevant_indices]
        inputs = inputs[relevant_indices]

        nlabels = ~labels
        pos_inv = nlabels.cumsum(-1)
        neg_inv = labels.sum() - labels.cumsum(-1)
        inversions = pos_inv*labels + neg_inv*nlabels
        assert (inversions > 0).all()
        # inv2 = compute_inversions(labels.numpy(), scores.numpy())
        # print(scores, labels, inversions, inv2)

        if dummy_forward :
            loss = torch.tensor(1.)
        else:
            pos_scores = scores[labels].reshape(-1,1)
            neg_scores = scores[~labels].reshape(1,-1)
            differences = -(pos_scores - neg_scores)
            losses = differences.clamp(min=0) 
            loss = losses.reshape(-1).sum()/npairs

        ctx.save_for_backward(weight, inputs, scores, labels, loss, npairs, inversions)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        weight, inputs, scores, labels, loss, npairs, inversions = ctx.saved_tensors

        if loss.item() == 0: 
            return torch.zeros_like(weight), None, None, None, None
                
        #inversions = torch.from_numpy(compute_inversions(labels.numpy(), scores.numpy()))

        ## use (possibly) sparse set of vectors with inversions
        # non_zero_coeffs = inversions > 0
        # inversions = inversions[non_zero_coeffs]
        # labels = labels[non_zero_coeffs]
        # inputs = inputs[non_zero_coeffs]
        signs = -(labels.float()*2 - 1.)
        coeffs = inversions.float() * signs / npairs
        delta = inputs.t() @ coeffs.reshape(-1) # matrix vector multiply
        # delta = -(coeffs.reshape(-1,1)*inputs).sum(axis=0)
        delta = delta.reshape(-1)
        assert delta.shape == weight.shape
        return delta.reshape(-1)    , None, None, None, None
    
    
class RankLoss(nn.Module):
    def __init__(self, w, margin: float, dummy: bool = False):
        super().__init__()
        self.w = nn.Parameter(data=F.normalize(w, dim=-1))
        self.margin = margin
        self.dummy = dummy
        
    def forward(self, dat, labels):
        # dat = F.normalize(dat,dim=-1)
        # with torch.no_grad():
        #     print(self.w.norm())
        #w = F.normalize(self.w, dim=-1)
        ## normalizing seems to affect gradients quite a bit even when 
        ## the norm is 1. so, don't normalize within gradient.
        loss = RankAndLoss.apply(self.w, dat, labels, torch.tensor(self.margin), self.dummy)
        return loss
    
class VecState:
    def __init__(self, w : np.ndarray, margin : float, opt_class, opt_params={}, renormalize=False):
        w = torch.from_numpy(w).type(torch.float32).reshape(-1)
        self.mod = RankLoss(w, margin, dummy=True)
        self.opt = opt_class([self.mod.w], **opt_params)
        print(self.opt)
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



