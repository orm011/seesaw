from .pairwise_rank_loss import *
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

def method2(w, data, labels,margin,):
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
            # assert torch.isclose(l1, l2), f'loss: {l1} vs {l2}' l2 is dummy.
            assert torch.isclose(g0, g2, atol=1e-6).all(), f'gradient: {g0} vs {g2}'
            assert torch.isclose(g1, g2, atol=1e-6).all(),f'gradient: {g1} vs {g2}'

#random_test_equiv(dim=100, batch_size=200)