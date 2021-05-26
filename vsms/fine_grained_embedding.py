import numpy as np
import pyroaring as pr
import os

from .search_loop_models import *
from .embeddings import *

def restrict_fine_grained(vec_meta, vec, indxs):
    assert vec_meta.shape[0] == vec.shape[0]
    indxs = np.array(indxs)
    assert (indxs[1:] > indxs[:-1]).all()
    mask = vec_meta.dbidx.isin(indxs)
    vec_meta = vec_meta[mask]
    vec = vec[mask]    
    old2new = dict(zip(indxs, np.arange(indxs.shape[0])))
    vec_meta = vec_meta.assign(dbidx=vec_meta.dbidx.map(lambda idx : old2new[idx]))
    
    assert vec_meta.shape[0] == vec.shape[0]
    return vec_meta.reset_index(drop=True), vec

class FineEmbeddingDB(object):
    """Structure holding a dataset,
     together with precomputed embeddings
     and (optionally) data structure
    """
    def __init__(self, raw_dataset : torch.utils.data.Dataset,
                 embedding : XEmbedding,
                 embedded_dataset : np.ndarray,
                 vector_meta : pd.DataFrame):
        self.raw = raw_dataset
        self.embedding = embedding
        all_indices = pr.BitMap()
        
        assert len(self.raw) == vector_meta.dbidx.unique().shape[0]
        assert embedded_dataset.shape[0] == vector_meta.shape[0]
        
        
        all_indices.add_range(0, len(self.raw))
        self.all_indices = pr.FrozenBitMap(all_indices)
        
        norms = np.linalg.norm(embedded_dataset, axis=1)[:,np.newaxis]
        self.embedded = embedded_dataset / (norms + 1e-5)
        self.vector_meta = vector_meta
    
    def __len__(self):
        return len(self.raw)

    def query(self, *, topk, mode, model=None, cluster_id=None, 
        vector=None, exclude=None, return_scores=False):
        if exclude is None:
            exclude = pr.BitMap([])        
        included_dbidx = pr.BitMap(self.all_indices).difference(exclude)
        
        vec_meta, vecs =  restrict_fine_grained(self.vector_meta, self.embedded, included_dbidx)
        
        if len(included_dbidx) == 0:
            if return_scores:
                return np.array([]),np.array([])
            else:
                return np.array([])

        if len(included_dbidx) <= topk:
            topk = len(included_dbidx)

        if vector is None and model is None:
            assert mode == 'random'
        elif vector is not None:
            assert mode in ['nearest', 'dot']
        elif model is not None:
            assert mode in ['model']
        else:
            assert False            

        if mode == 'nearest':
            scores = sklearn.metrics.pairwise.cosine_similarity(vector, vecs)
            scores = scores.reshape(-1)
        elif mode == 'dot':
            scores = vecs @ vector.reshape(-1)
        elif mode == 'random':
            scores = np.random.randn(vecs.shape[0])
        elif mode == 'model':
            with torch.no_grad():
                scores = model.forward(torch.from_numpy(vecs))
                scores = scores.numpy()[:,1]

        dbscores = (vec_meta
                        .assign(score=scores)[['dbidx', 'score']]
                        .groupby('dbidx').score.max() # max pooled...
                        .sort_values(ascending=False)).reset_index()
        
        scores = dbscores.score.iloc[:topk].values
        maxpos = dbscores.dbidx.iloc[:topk].values
        dbidxs = np.array(included_dbidx)[maxpos]

        ret = dbidxs
        assert ret.shape[0] == scores.shape[0]
        sret = pr.BitMap(ret)
        assert len(sret) == ret.shape[0]  # no repeats
        assert ret.shape[0] == topk  # return quantity asked, in theory could be less
        assert sret.intersection_cardinality(exclude) == 0  # honor exclude request

        if return_scores:
            return ret, scores
        else:
            return ret

import math
def resize_to_grid(base_size):
    """
    makes a transform that increases size and crops an image and any boxes so that the output meets the following 
        1. size of each side is least equal to base size
        2. base size fits an integer number of times along both dimensions using a stride of base_size//2
            the shorter side gets magnified, and the longer side gets cropped
        3. only tested with base_size = 224, what clip requires
        4. boxes assocated with image are also transformed to match the new image 
        5. each image is transformed differently dependig on its dimensions
    """
    def fun(im,boxes=None):
        hsize = base_size//2
        (w,h) = im.size
        sz = min(w,h)
        
        ## edge case:  side < 224 => side == 224.
        #max(2,math.ceil(sz/hsize))
        round_up = (math.ceil(sz/hsize)*hsize)
        scale_factor = max(base_size, round_up)/sz
        target_h = int(math.ceil(scale_factor*h))
        target_w = int(math.ceil(scale_factor*w))
        assert target_h >= base_size
        assert target_w >= base_size

        htrim = (target_h % hsize)
        wtrim = (target_w % hsize)
        crop_h = target_h - htrim
        crop_w = target_w - wtrim
        
        tx = T.Compose([T.Resize((target_h,target_w), interpolation=PIL.Image.BICUBIC),
                        T.CenterCrop((crop_h, crop_w))])
                  
        assert crop_h >= base_size
        assert crop_w >= base_size
        assert crop_h % hsize == 0
        assert crop_w % hsize == 0
        
        if boxes is not None:
            box0 = boxes
            sf = scale_factor
            box0 = box0.assign(x1=np.clip(box0.x1*sf - wtrim/2, 0, crop_w), 
                               x2=np.clip(box0.x2*sf - wtrim/2, 0, crop_w),
                               y1=np.clip(box0.y1*sf - htrim/2, 0, crop_h),
                               y2=np.clip(box0.y2*sf - htrim/2, 0, crop_h))
            return tx(im), box0
        else:
            return tx(im)
        
    return fun


def nearest_j(midx, width, base_size):
    hsize = base_size//2
    nlines = (width- hsize)//hsize
    ## draw diagram to see why we subtract.
    midx = midx - hsize/2
    midx = np.clip(midx, a_min=0, a_max=width)
    next_line_idx = midx//hsize
    next_line_idx = np.clip(next_line_idx, a_min=0,a_max=nlines-1)
    return next_line_idx.astype('int')

def nearest_ij(box, im, base_size):
    xs = (box.x1 + box.x2)/2
    jjs = nearest_j(xs, im.width, base_size=224).values
    
    ys = (box.y1 + box.y2)/2
    iis = nearest_j(ys, im.height, base_size=224).values
    return pd.DataFrame({'i':iis, 'j':jjs}, index=box.index)

def x2i(start, end, total, base_size=224):
    ## return i,j s for squares with that position...
    hsize = base_size//2
    max_i = (total - hsize)//hsize - 1 # stride of base_size//2, last stride doesn't count.
    i1 = np.clip(start//hsize-1,0,max_i).astype('int') # easy case
    i2 = np.clip(end//hsize,0,max_i).astype('int') 
    # this is correct except for the last one: in that case it should be the previous one...
    return i1,i2+1

def box2ij(box, im, base_size):
    (w,h) = im.size
    i1,i2 = x2i(start=box.y1,end=box.y2, total=h, base_size=base_size)
    j1,j2 = x2i(start=box.x1,end=box.x2, total=w, base_size=base_size)
    return pd.DataFrame({'i1':i1, 'i2':i2, 'j1':j1, 'j2':j2})

def get_pos_negs(box,im,vec_meta):
    """
    For a given image im, and a list of boxes (dataframe)
    and metadata of image vectors, compute 
     1. vectors of image chunks that do not overlap at all
     2. vectors of chunks nearest to box center.
    """
    ijs = box2ij(box, im, base_size=224)
    nearest_ijs = nearest_ij(box, im, base_size=224)
    tmp_meta = vec_meta#vec_meta[(vec_meta.dbidx == )]
    
    if box.shape[0] == 0:
        neg_idxs = pr.BitMap(tmp_meta.index.values)
        pos_idxs = pr.BitMap()
        return pos_idxs, neg_idxs
    
    
    negatives = [] 
    centers = []
    for tup,ctup in zip(ijs.itertuples(), nearest_ijs.itertuples()):
        overlap_ijs = ((tmp_meta.iis.between(tup.i1, tup.i2-1) & (tmp_meta.jjs.between(tup.j1, tup.j2 - 1))))
        negs = tmp_meta[~overlap_ijs] # no overlap whatsoever
        negatives.append(pr.BitMap(negs.index))
        cent = tmp_meta[(tmp_meta.iis == ctup.i) & (tmp_meta.jjs == ctup.j)]
        centers.append(pr.BitMap(cent.index))

    neg_idxs = pr.BitMap.intersection(*negatives)
    pos_idxs = pr.BitMap.union(*centers)
    return pos_idxs, neg_idxs

def get_pos_negs_all(dbidxs, ds, vec_meta):
    pos = []
    neg = []
    for idx in dbidxs:
        im,box = ds[idx]
        p, n = get_pos_negs(box,im,vec_meta[vec_meta.dbidx == idx])
        pos.append(p)
        neg.append(n)

    posidxs = pr.BitMap.union(*pos)
    negidxs = pr.BitMap.union(*neg)
    return posidxs, negidxs