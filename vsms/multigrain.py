import numpy as np
import pandas as pd
from .dataset_tools import *
from torch.utils.data import DataLoader
import math
from tqdm.auto import tqdm
import torch
from .embeddings import make_clip_transform, ImTransform, XEmbedding
from .vloop_dataset_loaders import get_class_ev
from .dataset_search_terms import  *
import pyroaring as pr

def postprocess_results(acc):
    flat_acc = {'iis':[], 'jjs':[], 'dbidx':[], 'vecs':[], 'zoom_factor':[], 'zoom_level':[]}
    flat_vecs = []
    for acc0,sf,dbidx,zl in acc:
        acc0 = acc0.squeeze(0)
        acc0 = acc0.transpose((1,2,0))

        iis, jjs = np.meshgrid(range(acc0.shape[0]), range(acc0.shape[1]), indexing='ij')
        #iis = iis.reshape(-1, acc0)
        iis = iis.reshape(-1)
        jjs = jjs.reshape(-1)
        acc0 = acc0.reshape(-1,acc0.shape[-1])
        imids = np.ones_like(iis)*dbidx
        zf = np.ones_like(iis)*(1./sf)
        zl = np.ones_like(iis)*zl

        flat_acc['iis'].append(iis)
        flat_acc['jjs'].append(jjs)
        flat_acc['dbidx'].append(imids)
        flat_acc['vecs'].append(acc0)
        flat_acc['zoom_factor'].append(zf)
        flat_acc['zoom_level'].append(zl)


    flat = {}
    for k,v in flat_acc.items():
        flat[k] = np.concatenate(v)

    vecs = flat['vecs']
    del flat['vecs']

    vec_meta = pd.DataFrame(flat)
    return vec_meta, vecs

def preprocess_ds(localxclip, ds, debug=False):
    txds = TxDataset(ds, tx=pyramid_tx(non_resized_transform(224)))
    acc = []
    if debug:
        num_workers=0
    else:
        num_workers=4
    for dbidx,tup in enumerate(tqdm(DataLoader(txds, num_workers=num_workers, shuffle=False, batch_size=1, collate_fn=lambda x : x), 
                     total=len(txds))):
        [(ims, sfs)] = tup
        for zoom_level,(im,sf) in enumerate(zip(ims,sfs),start=1):
            accs= localxclip.from_image(preprocessed_image=im, pooled=False)
            acc.append((accs, sf, dbidx, zoom_level))

    return postprocess_results(acc)

def pyramid_centered(im,i,j):
    cy=(i+1)*112.
    cx=(j+1)*112.
    scales = [112,224,448]
    crs = []
    w,h = im.size
    for s in scales:
        tup = (np.clip(cx-s,0,w), np.clip(cy-s,0,h), np.clip(cx+s,0,w), np.clip(cy+s,0,h))
        crs.append(im.crop(tup))
    return crs

def zoom_out(im, factor=.5, abs_min=224):
    """
        returns image one zoom level out, and the scale factor used
    """
    w,h=im.size
    mindim = min(w,h)
    target_size = max(math.floor(mindim*factor), abs_min)
    if target_size * math.sqrt(factor) <= abs_min: # if the target size is almost as large as the image, 
        # jump to that scale instead
        target_size = abs_min
    
    target_factor = target_size/mindim
    target_w = max(math.floor(w*target_factor),224) # corrects any rounding effects that make the size 223
    target_h = max(math.floor(h*target_factor),224)
    
    im1 = im.resize((target_w, target_h))
    assert min(im1.size) >= abs_min
    return im1, target_factor

def pyramid(im, factor=.5, abs_min=224):
    ims = []
    factors = [1.]
    while True:
        im, sf = zoom_out(im)
        ims.append(im)
        factors.append(sf*factors[-1])
        if min(im.size) == abs_min:
            break
            
    return ims, factors[1:]

def trim_edge(target_divisor=112):
    def fun(im1):
        w1,h1 = im1.size
        spare_h = h1 % target_divisor
        spare_w = w1 % target_divisor
        im1 = im1.crop((0,0,w1-spare_w, h1-spare_h))
        return im1
        
    return fun

def non_resized_transform(base_size):
    return ImTransform(visual_xforms=[trim_edge(base_size//2), lambda image: image.convert("RGB")],
                       tensor_xforms=[T.ToTensor(),
                                      T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                (0.26862954, 0.26130258, 0.27577711)),
                                        lambda x : x.type(torch.float16)])

def pyramid_tx(tx):
    def fn(im):
        ims,sfs= pyramid(im)
        ppims = []
        for im in ims:
            ppims.append(tx(im))

        return ppims, sfs
    return fn

def augment_score(db,tup,qvec):
    im = db.raw[tup.dbidx]
    ims = pyramid(im, tup.iis, tup.jjs)
    tx = make_clip_transform(n_px=224, square_crop=True)
    vecs = []
    for im in ims:
        pim = tx(im)
        emb = db.embedding.from_image(preprocessed_image=pim.float())
        emb = emb/np.linalg.norm(emb, axis=-1)
        vecs.append(emb)
        
    vecs = np.concatenate(vecs)
    #print(np.linalg.norm(vecs,axis=-1))
    augscore = (vecs @ qvec.reshape(-1)).mean()
    return augscore

import torchvision.ops
#torchvision.ops.box_iou()

def box_iou(tup, boxes):
    b1 = torch.from_numpy(np.stack([tup.x1.values, tup.y1.values, tup.x2.values, tup.y2.values], axis=1))
    bxdata = np.stack([boxes.x1.values, boxes.y1.values, boxes.x2.values,boxes.y2.values], axis=1)
    b2 = torch.from_numpy(bxdata)
    ious = torchvision.ops.box_iou(b1, b2)
    return ious.numpy()

def augment_score2(db,tup,qvec,vec_meta,vecs, rw_coarse=1.):
    vec_meta = vec_meta.reset_index(drop=True)    
    ious = box_iou(tup, vec_meta)
    vec_meta = vec_meta.assign(iou=ious.reshape(-1))
    max_boxes = vec_meta.groupby('zoom_level').iou.idxmax()
    max_boxes = max_boxes.sort_index(ascending=True) # largest zoom level (zoomed out) goes last 
    relevant_meta = vec_meta.iloc[max_boxes]
    relevant_iou = relevant_meta.iou > 0 #there should be at least some overlap for it to be relevant
    max_boxes = max_boxes[relevant_iou.values]
    sc = (vecs[max_boxes] @ qvec.reshape(-1,1))
    
    ws = np.ones_like(sc)
    ws[-1] = ws[-1]*rw_coarse
    fsc = ws.reshape(-1) @ sc.reshape(-1)
    fsc = fsc/ws.sum()
    return fsc

def get_boxes(vec_meta):
    y1 = vec_meta.iis*112
    y2 = y1 + 224
    x1 = vec_meta.jjs*112
    x2 = x1 + 224
    factor = vec_meta.zoom_factor
    boxes = vec_meta.assign(**{'x1':x1*factor, 'x2':x2*factor, 'y1':y1*factor, 'y2':y2*factor})[['x1','y1','x2','y2']]
    return boxes

def makedb(evs, dataset, category):
        ev,_ = get_class_ev(evs[dataset], category=category)
        return ev, AugmentedDB(raw_dataset=ev.image_dataset, embedding=ev.embedding, embedded_dataset=ev.fine_grained_embedding,
               vector_meta=ev.fine_grained_meta)

def try_augment(localxclip, evs, dataset, category, cutoff=40, rel_weight_coarse=1.):
    ev,db = makedb(evs, dataset, category)
    qvec = localxclip.from_raw(search_terms[dataset].get(category,category))
    qvec = qvec/np.linalg.norm(qvec)
    idxs = db.query(vector=qvec, topk=10, shortlist_size=cutoff, rel_weight_coarse=rel_weight_coarse)
    return db.raw.show_images(idxs, ev.query_ground_truth[category][idxs])


def get_pos_negs_all_v2(dbidxs, ds, vec_meta):
    idxs = pr.BitMap(dbidxs)
    relvecs = vec_meta[vec_meta.dbidx.isin(idxs)]
    
    pos = []
    neg = []
    for idx in dbidxs:
        acc_vecs = relvecs[relvecs.dbidx == idx]
        acc_boxes = get_boxes(acc_vecs)
        label_boxes = ds[idx]
        ious = box_iou(label_boxes, acc_boxes)
        total_iou = ious.sum(axis=0) 
        negatives = total_iou == 0
        negvec_positions = acc_vecs.index[negatives].values

        # get the highest iou positives for each 
        max_ious_id = np.argmax(ious,axis=1)
        max_ious = np.max(ious, axis=1)
        
        pos_idxs = pr.BitMap(max_ious_id[max_ious > 0])
        
        
#         breakpoint()
#         #max_ious_id[valmax > 0]        
#         positives = total_iou > .2 # eg. two granularities one within the other give you .25.
        posvec_positions = acc_vecs.index[pos_idxs].values
        pos.append(posvec_positions)
        neg.append(negvec_positions)
        
    posidxs = pr.BitMap(np.concatenate(pos))
    negidxs = pr.BitMap(np.concatenate(neg))
    return posidxs, negidxs

# def get_pos_negs_all_v2(dbidxs, ds, vec_meta):
#     idxs = pr.BitMap(dbidxs)
#     relvecs = vec_meta[vec_meta.dbidx.isin(idxs)]
    
#     pos = []
#     neg = []
#     for idx in dbidxs:
#         acc_vecs = relvecs[relvecs.dbidx == idx]
#         acc_boxes = get_boxes(acc_vecs)
#         label_boxes = ds[idx]
#         ious = box_iou(label_boxes, acc_boxes)
#         total_iou = ious.sum(axis=0) 
#         negatives = total_iou == 0
#         negvec_positions = acc_vecs.index[negatives].values

#         # get the highest iou positives for each 

#         positives = total_iou > .2 # eg. two granularities one within the other give you .25.
#         posvec_positions = acc_vecs.index[positives].values
        
#         pos.append(posvec_positions)
#         neg.append(negvec_positions)
        
#     posidxs = pr.BitMap(np.concatenate(pos))
#     negidxs = pr.BitMap(np.concatenate(neg))
#     return posidxs, negidxs


class AugmentedDB(object):
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
        self.embedded = embedded_dataset
        self.vector_meta = vector_meta.assign(**get_boxes(vector_meta))

        all_indices = pr.BitMap()
        assert len(self.raw) == vector_meta.dbidx.unique().shape[0]
        assert embedded_dataset.shape[0] == vector_meta.shape[0]
        all_indices.add_range(0, len(self.raw))
        self.all_indices = pr.FrozenBitMap(all_indices)
        
        #norms = np.linalg.norm(embedded_dataset, axis=1)[:,np.newaxis]
        # if not np.isclose(norms,0).all():
        #     print('warning: embeddings are not normalized?', norms.max())
    
    def __len__(self):
        return len(self.raw)

    def _query_prelim(self, *, vector, topk,  zoom_level, exclude=None):
        if exclude is None:
            exclude = pr.BitMap([])

        included_dbidx = pr.BitMap(self.all_indices).difference(exclude)
        vec_meta = self.vector_meta
        vecs = self.embedded  # = restrict_fine_grained(self.vector_meta, self.embedded, included_dbidx)
        
        # breakpoint()
        vec_meta = vec_meta.reset_index(drop=True)
        vec_meta = vec_meta[(~vec_meta.dbidx.isin(exclude))]
        if zoom_level is not None:
            vec_meta = vec_meta[vec_meta.zoom_level == zoom_level]

        vecs = vecs[vec_meta.index.values]

        if len(included_dbidx) == 0:
            return np.array([])

        if len(included_dbidx) <= topk:
            topk = len(included_dbidx)

        scores = vecs @ vector.reshape(-1)
        topscores = vec_meta.assign(score=scores).sort_values('score', ascending=False)
        # return topk unique images.
        seen_idx = set()
        for i,idx in enumerate(topscores.dbidx.values):
            seen_idx.add(idx)
            if len(seen_idx) == topk:
                break

        topscores = topscores.iloc[:i]
        assert topscores.dbidx.unique().shape[0] <= topk
        return topscores
        
    def query(self, *, vector, topk, mode='dot', exclude=None, shortlist_size=None, rel_weight_coarse=1):
        if shortlist_size is None:
            shortlist_size = topk*5
        
        db = self
        qvec=vector
        scmeta = self._query_prelim(vector=qvec, topk=shortlist_size, zoom_level=None, exclude=exclude)

        scs = []    
        for i,tup in enumerate(scmeta.itertuples()):
            relmeta = db.vector_meta[db.vector_meta.dbidx == tup.dbidx]
            relvecs = db.embedded[relmeta.index.values]
            tup = scmeta.iloc[i:i+1]
            sc = augment_score2(db, tup, qvec, relmeta, relvecs, rw_coarse=rel_weight_coarse)
            scs.append(sc)

        final = scmeta.assign(aug_score=np.array(scs))
        agg = final.groupby('dbidx').aug_score.max().reset_index().sort_values('aug_score', ascending=False)
        idxs = agg.dbidx.iloc[:topk]
        return idxs