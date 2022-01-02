from ray.data.extensions import TensorArray

import torchvision
from torchvision import transforms as T

import numpy as np
import pandas as pd
from .dataset_tools import *
from torch.utils.data import DataLoader
import math
from tqdm.auto import tqdm
import torch
from .query_interface import *

from .embeddings import make_clip_transform, ImTransform, XEmbedding
from .vloop_dataset_loaders import get_class_ev
from .dataset_search_terms import  *
import pyroaring as pr
from operator import itemgetter
import PIL

import math
import annoy


def _postprocess_results(acc):
    flat_acc = {'iis':[], 'jjs':[], 'dbidx':[], 'vecs':[], 'zoom_factor':[], 'zoom_level':[]}
    flat_vecs = []

    #{'accs':accs, 'sf':sf, 'dbidx':dbidx, 'zoom_level':zoom_level}
    for item in acc:
        acc0,sf,dbidx,zl = itemgetter('accs', 'sf', 'dbidx', 'zoom_level')(item)
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
    vecs = vecs.astype('float32')
    vecs = vecs/(np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-6)
    vec_meta = vec_meta.assign(file_path=item['file_path'])

    vec_meta = vec_meta.assign(vectors=TensorArray(vecs))
    return vec_meta

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

    return _postprocess_results(acc)

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

def zoom_out(im : PIL.Image, factor=.5, abs_min=224):
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


def rescale(im, scale, min_size):
    (w,h) = im.size
    target_w = max(math.floor(w*scale),min_size)
    target_h = max(math.floor(h*scale),min_size)
    return im.resize(size=(target_w, target_h), resample=PIL.Image.BILINEAR)

def pyramid(im, factor=.71, abs_min=224):
    ## if im size is less tha the minimum, expand image to fit minimum
    ## try following: orig size and abs min size give you bounds
    assert factor < 1.
    factor = 1./factor
    size = min(im.size)
    end_size = abs_min
    start_size = max(size, abs_min)

    start_scale = start_size/size
    end_scale = end_size/size

    ## adjust start scale
    ntimes = math.ceil(math.log(start_scale/end_scale)/math.log(factor))
    start_size = math.ceil(math.exp(ntimes*math.log(factor) + math.log(abs_min)))
    start_scale = start_size/size
    factors = np.geomspace(start=start_scale, stop=end_scale, num=ntimes+1, endpoint=True).tolist()
    ims = []
    for sf in factors:
        imout = rescale(im, scale=sf, min_size=abs_min)
        ims.append(imout)

    assert len(ims) > 0
    assert min(ims[0].size) >= abs_min
    assert min(ims[-1].size) == abs_min
    return ims, factors

def trim_edge(target_divisor=112):
    def fun(im1):
        w1,h1 = im1.size
        spare_h = h1 % target_divisor
        spare_w = w1 % target_divisor
        im1 = im1.crop((0,0,w1-spare_w, h1-spare_h))
        return im1
        
    return fun

class TrimEdge:
    def __init__(self, target_divisor=112):
        self.target_divisor = target_divisor

    def __call__(self, im1):
        w1,h1 = im1.size
        spare_h = h1 % self.target_divisor
        spare_w = w1 % self.target_divisor
        im1 = im1.crop((0,0,w1-spare_w, h1-spare_h))
        return im1

def torgb(image):
    return image.convert('RGB')

def tofloat16(x):
    return x.type(torch.float16)

def non_resized_transform(base_size):
    return ImTransform(visual_xforms=[torgb],
                       tensor_xforms=[T.ToTensor(),
                                      T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                (0.26862954, 0.26130258, 0.27577711)),
                                                #tofloat16
                                                ])


class PyramidTx:
    def __init__(self, tx, factor, min_size):
        self.tx = tx
        self.factor = factor
        self.min_size = min_size

    def __call__(self, im):
        ims,sfs= pyramid(im, factor=self.factor, abs_min=self.min_size)
        ppims = []
        for im in ims:
            ppims.append(self.tx(im))

        return ppims, sfs

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
    if 'x1' in vec_meta.columns:
        return vec_meta[['x1', 'x2', 'y1', 'y2']]
    
    y1 = vec_meta.iis*112
    y2 = y1 + 224
    x1 = vec_meta.jjs*112
    x2 = x1 + 224
    factor = vec_meta.zoom_factor
    boxes = vec_meta.assign(**{'x1':x1*factor, 'x2':x2*factor, 'y1':y1*factor, 'y2':y2*factor})[['x1','y1','x2','y2']]
    boxes = boxes.astype('float32') ## multiplication makes this type double but this is too much.
    return boxes

def makedb(evs, dataset, category):
        ev,_ = get_class_ev(evs[dataset], category=category)
        return ev, MultiscaleIndex(images=ev.image_dataset, embedding=ev.embedding, vectors=ev.fine_grained_embedding,
               vector_meta=ev.fine_grained_meta, vec_index=ev.vec_index)


def get_pos_negs_all_v2(dbidxs, box_dict, vec_meta):
    idxs = pr.BitMap(dbidxs)
    relvecs = vec_meta[vec_meta.dbidx.isin(idxs)]
    
    pos = []
    neg = []
    for idx in dbidxs:
        acc_vecs = relvecs[relvecs.dbidx == idx]
        acc_boxes = get_boxes(acc_vecs)
        label_boxes = box_dict[idx]
        ious = box_iou(label_boxes, acc_boxes)
        total_iou = ious.sum(axis=0) 
        negatives = total_iou == 0
        negvec_positions = acc_vecs.index[negatives].values

        # get the highest iou positives for each 
        max_ious_id = np.argmax(ious,axis=1)
        max_ious = np.max(ious, axis=1)
        
        pos_idxs = pr.BitMap(max_ious_id[max_ious > 0])
        # if label_boxes.shape[0] > 0: # some boxes are size 0 bc. of some bug in the data, so don't assert here.
        #     assert len(pos_idxs) > 0

        posvec_positions = acc_vecs.index[pos_idxs].values
        pos.append(posvec_positions)
        neg.append(negvec_positions)
        
    posidxs = pr.BitMap(np.concatenate(pos))
    negidxs = pr.BitMap(np.concatenate(neg))
    return posidxs, negidxs

def build_index(vecs, file_name):
    t = annoy.AnnoyIndex(512, 'dot') 
    for i in range(len(vecs)):
        t.add_item(i, vecs[i])
    t.build(n_trees=100) # tested 100 on bdd, works well, could do more.
    t.save(file_name)
    u = annoy.AnnoyIndex(512, 'dot')
    u.load(file_name) # verify can load.
    return u

class MultiscaleIndex(AccessMethod):
    """implements a two stage lookup
    """
    def __init__(self, *,
                 images : torch.utils.data.Dataset,
                 embedding : XEmbedding,
                 vectors : np.ndarray,
                 vector_meta : pd.DataFrame,
                 vec_index =  None):

        self.images = images
        self.embedding = embedding
        self.vectors = vectors
        self.vector_meta = vector_meta
        self.vec_index = vec_index

        all_indices = pr.BitMap()
        assert len(self.images) >= vector_meta.dbidx.unique().shape[0]
        assert vectors.shape[0] == vector_meta.shape[0]
        all_indices.add_range(0, len(self.images))
        self.all_indices = pr.FrozenBitMap(all_indices)

    def string2vec(self, string : str):
        init_vec = self.embedding.from_string(string=string)
        init_vec = init_vec/np.linalg.norm(init_vec)
        return init_vec
    
    def __len__(self):
        return len(self.images)

    def _query_prelim(self, *, vector, topk,  zoom_level, exclude=None, startk=None):
        if exclude is None:
            exclude = pr.BitMap([])

        included_dbidx = pr.BitMap(self.all_indices).difference(exclude)
        vec_meta = self.vector_meta
        
        if len(included_dbidx) == 0:
            print('no dbidx included')
            return [], [],[]

        if len(included_dbidx) <= topk:
            topk = len(included_dbidx)
            
        ## want to return proposals only for images we have not seen yet...
        ## but library does not allow this...
        ## guess how much we need... and check
        
        def get_nns_with_pynn_index():
            i = 0
            try_n = (len(exclude) + topk)*10 # guess of how many to use
            while True:
                if i > 1:
                    print('warning, we are looping too much. adjust initial params?')

                idxs, scores = self.vec_index.query(vector.reshape(1,-1), k=try_n, epsilon=.2)
                idxs = np.array(idxs).astype('int').reshape(-1)
                scores = np.array(scores).reshape(-1) 
                # search_k = int(np.clip(try_n * 100, 50000, 500000)) 
                # search_k is a very important param for accuracy do not reduce below 30k unless you have a 
                # really good reason. the upper limit is 
                # idxs, scores = self.vec_index.get_nns_by_vector(vector.reshape(-1), n=try_n, 
                #                                                 search_k=search_k,
                #                                                 include_distances=True)
                #breakpoint()
                found_idxs = pr.BitMap(vec_meta.dbidx.values[idxs])
                if len(found_idxs.difference(exclude)) >= topk:
                    break
                
                try_n = try_n*2 # double guess
                i+=1

            return idxs, scores


        def get_nns_by_vector_approx():
            i = 0
            try_n = (len(exclude) + topk)*3
            while True:
                if i > 1:
                    print('warning, we are looping too much. adjust initial params?')

                #search_k = int(np.clip(try_n * 100, 50000, 500000)) 
                # search_k is a very important param for accuracy do not reduce below 30k unless you have a 
                # really good reason. the upper limit is 
                idxs, scores = self.vec_index.get_nns_by_vector(vector.reshape(-1), n=try_n, 
                 #                                               search_k=search_k,
                                                                include_distances=True)


                found_idxs = pr.BitMap(vec_meta.dbidx.values[idxs])
                if len(found_idxs.difference(exclude)) >= topk:
                    break
                
                try_n = try_n*2
                i+=1

            return np.array(idxs).astype('int'), np.array(scores)


        def get_nns(startk, topk):
            i = 0
            deltak = topk*100
            while True:
                if i > 1:
                    print('warning, we are looping too much. adjust initial params?')

                idxs,scores = self.vec_index.query(vector, top_k=startk + deltak)
                found_idxs = pr.BitMap(vec_meta.dbidx.values[idxs])

                newidxs = found_idxs.difference(exclude)
                if len(newidxs) >= topk:
                    break
                
                deltak = deltak*2
                i+=1

            return idxs, scores

        def get_nns_by_vector_exact():
            scores = self.vectors @ vector.reshape(-1)
            scorepos = np.argsort(-scores)
            return scorepos, scores[scorepos]

        if self.vec_index is not None:
            idxs, scores  = get_nns(startk, topk)
        else:
            idxs, scores = get_nns_by_vector_exact()

        # work only with the two columns here bc dataframe can be large
        topscores = vec_meta[['dbidx']].iloc[idxs]
        topscores = topscores.assign(score=scores)
        allscores = topscores
        
        newtopscores = topscores[~topscores.dbidx.isin(exclude)]
        scoresbydbidx = newtopscores.groupby('dbidx').score.max().sort_values(ascending=False)
        score_cutoff = scoresbydbidx.iloc[topk-1] # kth largest score
        newtopscores = newtopscores[newtopscores.score >=  score_cutoff]

        # newtopscores = newtopscores.sort_values(ascending=False)
        nextstartk = (allscores.score >= score_cutoff).sum()
        nextstartk  = math.ceil(startk*.8 + nextstartk*.2) # average to estimate next
        candidates =  pr.BitMap(newtopscores.dbidx)
        assert len(candidates) >= topk
        assert candidates.intersection_cardinality(exclude) == 0
        return newtopscores.index.values, candidates, allscores, nextstartk
        
    def query(self, *, vector, topk, exclude=None, rel_weight_coarse=1, startk=None, **kwargs):
#        print('ignoring extra args:', kwargs)
        shortlist_size = topk*5
        
        if startk is None:
            startk = len(exclude)*10

        db = self
        qvec=vector
        meta_idx, candidate_id, allscores, nextstartk = self._query_prelim(vector=qvec, topk=shortlist_size, 
                                            zoom_level=None, exclude=exclude, startk=startk)

        fullmeta = self.vector_meta[self.vector_meta.dbidx.isin(candidate_id)]
        fullmeta = fullmeta.assign(**get_boxes(fullmeta))

        scmeta = self.vector_meta.iloc[meta_idx]
        scmeta = scmeta.assign(**get_boxes(scmeta))
        nframes = len(candidate_id)
        dbidxs = np.zeros(nframes)*-1
        dbscores = np.zeros(nframes)

        ## for each frame, compute augmented scores for each tile and record max
        for i,(dbidx,frame_vec_meta) in enumerate(scmeta.groupby('dbidx')):
            dbidxs[i] = dbidx
            relmeta = fullmeta[fullmeta.dbidx == dbidx] # get metadata for all boxes in frame.
            relvecs = db.vectors[relmeta.index.values]
            boxscs = np.zeros(frame_vec_meta.shape[0])
            for j in range(frame_vec_meta.shape[0]):
                tup = frame_vec_meta.iloc[j:j+1]
                boxscs[j] = augment_score2(db, tup, qvec, relmeta, relvecs, rw_coarse=rel_weight_coarse)

            dbscores[i] = np.max(boxscs)

        topkidx = np.argsort(-dbscores)[:topk]
        return dbidxs[topkidx].astype('int'), nextstartk # return fullmeta 

    def new_query(self):
        return BoxFeedbackQuery(self)

class BoxFeedbackQuery(InteractiveQuery):
    def __init__(self, db):
        super().__init__(db)
        self.acc_pos = []
        self.acc_neg = []

    def getXy(self, idxbatch, box_dict):
        batchpos, batchneg = get_pos_negs_all_v2(idxbatch, box_dict, self.db.vector_meta)
 
        ## we are currently ignoring these positives
        self.acc_pos.append(batchpos)
        self.acc_neg.append(batchneg)

        pos = pr.BitMap.union(*self.acc_pos)
        neg = pr.BitMap.union(*self.acc_neg)

        allpos = self.db.vectors[pos]
        Xt = np.concatenate([allpos, self.db.vectors[neg]])
        yt = np.concatenate([np.ones(len(allpos)), np.zeros(len(neg))])

        return Xt,yt
        # not really valid. some boxes are area 0. they should be ignored.but they affect qgt
        # if np.concatenate(acc_results).sum() > 0:
        #    assert len(pos) > 0