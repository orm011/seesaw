import torch.nn as nn
import torch.nn.functional as F
import clip
from .dataset_search_terms import category2query
import numpy as np
import pandas as pd
from .query_interface import AccessMethod
from IPython.display import display

class StringEncoder(object):
    def __init__(self):
        variant ="ViT-B/32"
        device='cpu'
        jit = False
        self.device = device
        model, preproc = clip.load(variant, device=device,  jit=jit)
        self.model = model
        self.preproc = preproc
        self.celoss = nn.CrossEntropyLoss(reduction='none')
        
    def encode_string(self, string):
        model = self.model.eval()
        with torch.no_grad():
            ttext = clip.tokenize([string])
            text_features = model.encode_text(ttext.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.detach().cpu().numpy()

def get_text_features(self, actual_strings, target_string):        
    s2id = {}
    sids = []
    s2id[target_string] = 0
    for s in actual_strings:
        if s not in s2id:
            s2id[s] = len(s2id)

        sids.append(s2id[s])

    strings = [target_string] + actual_strings
    ustrings = list(s2id)
    stringids = torch.tensor([s2id[s] for s in actual_strings], dtype=torch.long).to(self.device)
    tstrings = clip.tokenize(ustrings)
    text_features = self.model.encode_text(tstrings.to(self.device))
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, stringids, ustrings
        
def forward(self, imagevecs, actual_strings, target_string):
    ## uniquify strings    
    text_features, stringids, ustrings = get_text_features(self, actual_strings, target_string)

    image_features = torch.from_numpy(imagevecs).type(text_features.dtype)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.to(self.device)
    scores = image_features @ text_features.t()
    
    assert scores.shape[0] == stringids.shape[0]
    return scores, stringids.to(self.device), ustrings

def forward2(self, imagevecs, actual_strings, target_string):
    text_features, stringids, ustrings = get_text_features(self, actual_strings, target_string)
    actual_vecs = text_features[stringids]
    sought_vec = text_features[0].reshape(1,-1)
    
    image_features = torch.from_numpy(imagevecs).type(text_features.dtype)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.to(self.device)

    search_score = image_features @ sought_vec.reshape(-1)
    confounder_score = (image_features * actual_vecs).sum(dim=1)
    return search_score, confounder_score
    
import torch.optim

def show_scores(se, vecs, actual_strings, target_string):
    with torch.no_grad():
        se.model.eval()
        scs,stids,rawstrs = forward(se, vecs, actual_strings, target_string=target_string)
    scdf = pd.DataFrame({st:col  for  st,col in zip(rawstrs,scs.cpu().numpy().transpose())})
    display(scdf.style.highlight_max(axis=1))

def get_feedback(idxbatch):
    strids = np.where(ev.query_ground_truth.iloc[idxbatch])[1]
    strs = ev.query_ground_truth.columns[strids]
    strs = [category2query(dataset='objectnet', cat=fbstr) for fbstr in strs.values]
    return strs

class Updater(object):
    def __init__(self, se, lr, rounds=1, losstype='hinge'):
        self.se = se
        self.losstype=losstype
        self.opt = torch.optim.AdamW([{'params': se.model.ln_final.parameters()},
                          {'params':se.model.text_projection},
#                          {'params':se.model.transformer.parameters(), 'lr':lr*.01}
                                     ], lr=lr, weight_decay=0.)
#        self.opt = torch.optim.Adam@([{'params': se.model.parameters()}], lr=lr)
        self.rounds = rounds
        
    def update(self, imagevecs, actual_strings, target_string):
        se = self.se
        se.model.train()
        losstype = self.losstype
        opt = self.opt
        margin = .3

        def opt_closure():
            opt.zero_grad()            
            if losstype=='ce':
                scores, stringids, rawstrs = forward(se, imagevecs, actual_strings, target_string)
                # breakpoint()
                iidx = torch.arange(scores.shape[0]).long()
                actuals = scores[iidx, stringids]
                midx = scores.argmax(dim=1)
                maxes = scores[iidx, midx]                
            elif losstype=='hinge':
                #a,b = forward2(se, imagevecs, actual_strings, target_string)
                scores, stringids, rawstrs = forward(se, imagevecs, actual_strings, target_string)
                # breakpoint()
                iidx = torch.arange(scores.shape[0]).long()
                maxidx = scores.argmax(dim=1)
                
                actual_score = scores[iidx, stringids].reshape(-1,1)
                #max_score = scores[iidx, maxidx]
                
                
                #target_score = scores[:,0]
                losses1 = F.relu(- (actual_score - scores - margin))
                #losses2 = F.relu(- (actual_score - target_score - margin))
                #losses = torch.cat([losses1, losses2])
                losses = losses1
            else:
                assert False
            loss = losses.mean()
            #print(loss.detach().cpu())
            loss.backward()

        for _ in range(self.rounds):
            opt.step(opt_closure)

def closure(idx : AccessMethod, qgt : pd.DataFrame, search_query, max_n, firsts, show_display=False, batch_size=10):
    sq = category2query('objectnet', search_query)
    se = StringEncoder()
    up = Updater(se, lr=.0001, rounds=1)
    bs = batch_size
    bfq = idx.new_query()
    tvecs = []
    dbidxs = []
    accstrs = []
    gts = []
    while True:
        tvec = se.encode_string(sq)
        tvecs.append(tvec)
        idxbatch, _ = bfq.query_stateful(mode='dot', vector=tvec, batch_size=bs)
        dbidxs.append(idxbatch)
        gtvals = qgt[search_query][idxbatch].values
        gts.append(gtvals)
        #vecs = ev.embedded_dataset[idxbatch]
        actual_strings = get_feedback(idxbatch)
        accstrs.extend(actual_strings)

        if show_display:
            display(actual_strings)
        if gtvals.sum() > 0 or len(accstrs) > max_n:
            break

    #     vcs = ev.embedded_dataset[idxbatch]
    #     astrs = actual_strings    
        vcs = ev.embedded_dataset[np.concatenate(dbidxs)]
        astrs = accstrs

        if show_display:
            show_scores(se, vcs, astrs, target_string=sq)
            
        up.update(vcs, actual_strings=astrs, target_string=sq)

        if show_display:
            show_scores(se, vcs, astrs, target_string=sq)


    frsts = np.where(np.concatenate(gts).reshape(-1))[0]
    if frsts.shape[0] == 0:
        firsts[search_query] = np.inf
    else:
        firsts[search_query] = frsts[0] + 1