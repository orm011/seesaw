#from .data_server import *
from .search_loop_models import *
from .search_loop_tools import *

import inspect
from .dataset_tools import *
from .vloop_dataset_loaders import EvDataset, get_class_ev
from .fine_grained_embedding import *
from .multigrain import *
from .cross_modal_db import EmbeddingDB
from .search_loop_models import adjust_vec, adjust_vec2
import numpy as np
import sklearn.metrics
import math
from .util import *

def vls_init_logger():
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)

def brief_format(ftpt):
    if math.isclose(ftpt, 0.):
        return '0'
    
    if math.isclose(ftpt,1.):
        return '1'

    if ftpt < 1.:
        exp = -math.floor(math.log10(abs(ftpt)))
        fmt_string = '{:.0%df}' % (exp + 1)
        dec = fmt_string.format(ftpt)    
    else:
        fmt_string = '{:.02f}'
        dec = fmt_string.format(ftpt)
        
    zstripped = dec.lstrip('0').rstrip('0')
    return zstripped.rstrip('.')


def times_format(ftpt):
    return brief_format(ftpt) + 'x'

def make_labeler(fmt_func):
    def fun(arrlike):
        return list(map(fmt_func, arrlike))
    return fun

import numpy as np

def readjust_interval(x1, x2, max_x):
    left_excess = -np.clip(x1,-np.inf,0)
    right_excess = np.clip(x2 - max_x, 0,np.inf)

    x1p = x1 + left_excess - right_excess
    x2p = x2 + left_excess - right_excess
    
    assert ((x2p - x1p) == (x2 - x1)).all()
    assert (x1p >= 0).all()
    assert (x2p <= max_x).all()
    return x1p, x2p

def random_seg_start(x1, x2, target_x, max_x, off_center_range, n=1):
    dist = x2 - x1
    assert (dist <= target_x).all()
    gap = (target_x - dist)
    start_offset = .5 + (np.random.rand(n) - .5)*off_center_range
    assert (start_offset >= 0.).all()
    assert (start_offset <= 1.).all()
    assert (abs(start_offset - .5) <= off_center_range).all()
    start = x1 - start_offset*gap
    end = start + target_x
    start, end = readjust_interval(start,end,max_x)
    return start, end

def add_clearance(x1,x2,max_x, clearance_ratio):
    cx = (x1 + x2)/2
    dx = x2 - x1
    diff = dx*clearance_ratio*.5
    return readjust_interval(cx - diff, cx + diff, max_x)

def add_box_clearance(b, max_x, max_y, clearance_ratio):
    x1,x2 = add_clearance(b.x1, b.x2, max_x, clearance_ratio)
    y1,y2 = add_clearance(b.y1, b.y2, max_y, clearance_ratio)
    return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2}

def random_container_box(b, scale_range=3.3, aspect_ratio_range=1.2, off_center_range=1., clearance=1.2, n=1):
    assert clearance >= aspect_ratio_range

    bw = b.x2 - b.x1
    bh = b.y2 - b.y1
    max_d = max(bw,bh)
    sc1 = b.im_height/max_d
    sc2 = b.im_width/max_d

    min_scale = min(sc1, sc2, clearance)
    clearance = min(min_scale, clearance)

    max_scale = min(sc1, sc2, scale_range*clearance) # don't do more than 3x
    max_scale = max(min_scale, max_scale)

    scale = np.exp(np.random.rand(n)*np.log(max_scale/min_scale))*min_scale
    # assert (scale >= clearance).all()
    # assert (scale <= scale_range).all()

    target_x = scale*max_d
    target_y = target_x
    assert (bw <= target_x).all()
    assert (bh <= target_y).all()
    
    lratio = 2*(np.random.rand(n) - .5)*np.log(aspect_ratio_range)
    ratio = np.exp(lratio/2)

    upper = math.sqrt(aspect_ratio_range)
    assert (ratio <= upper).all()
    assert (ratio >= 1/upper).all()
        
    target_y = target_y*ratio
    target_x = target_x/ratio #np.ones_like(ratio)
    start_x, end_x = random_seg_start(b.x1, b.x2, target_x, b.im_width, off_center_range=off_center_range, n=n)
    start_y, end_y = random_seg_start(b.y1, b.y2, target_y, b.im_height, off_center_range=off_center_range, n=n)
    
    assert (start_x <= b.x1).all()
    assert (end_x >= b.x2).all()
    
    return pd.DataFrame({'x1':start_x, 'x2': end_x, 'y1':start_y, 'y2':end_y})

def randomly_extended_crop(im, box, scale_range, aspect_ratio_range, off_center_range, clearance, n):
    rbs = random_container_box(box, scale_range, aspect_ratio_range, off_center_range, clearance, n=n)
    crs = []
    for cb in rbs.itertuples():
        cr = im.crop((cb.x1, cb.y1, cb.x2, cb.y2))
        crs.append(cr)
    return cr


def run_loop6(*, ev :EvDataset, category, qstr, interactive, warm_start, n_batches, batch_size, minibatch_size, 
              learning_rate, max_examples, num_epochs, loss_margin, 
              tqdm_disabled:bool, granularity:str,
               model_type='logistic', solver_opts={}, 
               **kwargs):         
    assert 'fine_grained' not in kwargs
    assert isinstance(granularity, str)
    # gvec = ev.fine_grained_embedding
    # gvec_meta = ev.fine_grained_meta
    ev0 = ev
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    allargs = {k:v for (k,v) in values.items() if k in args and k not in ['ev', 'gvec', 'gvec_meta']}        

    ev, class_idxs = get_class_ev(ev0, category, boxes=True)
    dfds =  DataFrameDataset(ev.box_data[ev.box_data.category == category], index_var='dbidx', max_idx=class_idxs.shape[0]-1)
    
    rsz = resize_to_grid(224)
    ds = TxDataset(dfds, tx=lambda tup : rsz(im=None, boxes=tup)[1])
    imds = TxDataset(ev.image_dataset, tx = lambda im : rsz(im=im, boxes=None)[0])

    if granularity == 'fine':
        vec_meta = ev.fine_grained_meta
        vecs = ev.fine_grained_embedding
        vec_meta = vec_meta[vec_meta.zoom_level == 0]
        vecs = vecs[vec_meta.index.values]
        vec_meta.reset_index(drop=True)
        hdb = FineEmbeddingDB(raw_dataset=ev.image_dataset, embedding=ev.embedding, 
            embedded_dataset=vecs, vector_meta=vec_meta)
    elif granularity == 'multi':
        # assert False, 'not using this at the moment'
        vec_meta = ev.fine_grained_meta
        vecs = ev.fine_grained_embedding
        # dbidxs = np.arange(len(ev)).astype('int')
        # vec_meta_coarse = pd.DataFrame({'iis': np.zeros_like(dbidxs), 'jjs':np.zeros_like(dbidxs), 'dbidx':dbidxs})
        # vec_meta_coarse = vec_meta_coarse.assign(scale='coarse')
        # vecs_coarse = ev.embedded_dataset
        # vec_meta = pd.concat([vec_meta_fine, vec_meta_coarse], ignore_index=True)
        # vecs = np.concatenate([vecs_fine, vecs_coarse])

        hdb = AugmentedDB(raw_dataset=ev.image_dataset, embedding=ev.embedding, 
            embedded_dataset=vecs, vector_meta=vec_meta)

    elif granularity == 'coarse':
        dbidxs = np.arange(len(ev)).astype('int')
        vec_meta = pd.DataFrame({'iis': np.zeros_like(dbidxs), 'jjs':np.zeros_like(dbidxs), 'dbidx':dbidxs})
        vecs = ev.embedded_dataset
        hdb = EmbeddingDB(raw_dataset=ev.image_dataset, embedding=ev.embedding,embedded_dataset=vecs)
    else:
        assert False

    bfq = BoxFeedbackQuery(hdb, batch_size=batch_size, auto_fill_df=None)
    rarr = ev.query_ground_truth[category]
    print(allargs, kwargs)        
            
    if qstr == 'nolang':
        init_vec=None
        init_mode = 'random'
    else:
        init_vec = ev.embedding.from_string(string=qstr)
        init_mode = 'dot'
    
    acc_pos = []
    acc_neg = []
    acc_indices = []
    acc_results = []
    acc_ranks = []
    total = 0
            
    gt = ev.query_ground_truth[category].values
    res = {'indices':acc_indices, 'results':acc_results}#, 'gt':gt.values.copy()}
    if init_vec is not None:
        init_vec = init_vec/np.linalg.norm(init_vec)
        tvec = init_vec#/np.linalg.norm(init_vec)
    else:
        tvec = None

    tmode = init_mode
    pos_vecs = []

    for i in tqdm(range(n_batches), leave=False, disable=tqdm_disabled):
        idxbatch = bfq.query_stateful(mode=tmode, vector=tvec, batch_size=batch_size)
        acc_indices.append(idxbatch)
        acc_results.append(gt[idxbatch])

        if interactive != 'plain':
            if granularity in ['fine', 'multi']:
                batchpos, batchneg = get_pos_negs_all_v2(idxbatch, ds, vec_meta)
                ### here, get all boxes. extract features. use those as positive vecs. (noindex)
                ## where are the boxes?
                # copy_locals('loopvars')#breakpoint()
                # breakpoint()
                # batchpos = np.array(batchpos)
                # batchneg = np.array(batchneg)

                # vm1 = vec_meta.iloc[batchpos].zoom_level == 0
                # vm2 = vec_meta.iloc[batchneg].zoom_level == 0
                # batchpos = pr.BitMap(batchpos[vm1.values])
                # batchneg = pr.BitMap(batchneg[vm2.values]) # vec_meta.iloc[batchpos].zoom_level == 0]
                acc_pos.append(batchpos)
                acc_neg.append(batchneg)

                pos = pr.BitMap.union(*acc_pos)
                neg = pr.BitMap.union(*acc_neg)
                # print('pos, neg', len(pos), len(neg))

                for idx in idxbatch:
                    boxes = ds[idx]
                    if boxes.shape[0] == 0:
                        continue
                    im = imds[idx]
                    for b in boxes.iteritems():
                        cr = randomly_extended_crop(im, b, scale_range=1., aspect_ratio_range=1., off_center_range=0., clearance=1.5, n=1)

                        ev.embedding.from_image(image=cr)

                Xt = np.concatenate([vecs[pos], vecs[neg]])
                yt = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
                ## TODO: run extractor on augmented boxes
                ## TODO: run it on nicely centered boxes to make sure


                if np.concatenate(acc_results).sum() > 0:
                    assert len(pos) > 0
            else:
                Xt = vecs[idxbatch]
                yt = gt[idxbatch]

            if (yt.shape[0] > 0) and (yt.max() > yt.min()):
                tmode = 'dot'
                if interactive == 'sklearn':
                    lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
                    lr.fit(Xt, yt)
                    tvec = lr.coef_.reshape(1,-1)        
                elif interactive == 'pytorch':
                    p = yt.sum()/yt.shape[0]
                    w = np.clip((1-p)/p, .1, 10.)

                    if model_type == 'logistic':
                        mod = PTLogisiticRegression(Xt.shape[1], learning_rate=learning_rate, C=0, 
                                                    positive_weight=w)
                        if warm_start == 'warm':
                            iv = torch.from_numpy(init_vec)
                            iv = iv / iv.norm()
                            mod.linear.weight.data = iv.type(mod.linear.weight.dtype)
                        elif warm_start == 'default':
                            pass

                        fit_reg(mod=mod, X=Xt.astype('float32'), y=yt.astype('float'), batch_size=minibatch_size)
                        tvec = mod.linear.weight.detach().numpy().reshape(1,-1)
                    elif model_type in ['cosine', 'multirank']:
                        for i in range(num_epochs):
                            tvec = adjust_vec(tvec, Xt, yt, learning_rate=learning_rate, 
                                                max_examples=max_examples, 
                                                minibatch_size=minibatch_size,
                                                loss_margin=loss_margin)
                    elif model_type == 'solver':
                        tvec = adjust_vec2(tvec, Xt, yt, **solver_opts)
                    else:
                        assert False, 'model type'

                else:
                    assert False
            else:
                # print('missing positives or negatives to do any training', yt.shape, yt.max(), yt.min())
                pass
            
    res['indices'] = [class_idxs[r] for r in res['indices']]
    tup = {**allargs, **kwargs}
    return (tup, res)


def ndcg_rank_score(ytrue, ordered_idxs):
    '''
        wraps sklearn.metrics.ndcg_score to grade a ranking given as an ordered array of k indices,
        rather than as a score over the full dataset.
    '''
    ytrue = ytrue.astype('float')
    # create a score that is consistent with the ranking.
    ypred = np.zeros_like(ytrue)
    fake_scores = np.arange(ordered_idxs.shape[0],0,-1)/ordered_idxs.shape[0]
    assert (fake_scores > 0).all()
    ypred[ordered_idxs] = fake_scores
    return sklearn.metrics.ndcg_score(ytrue.reshape(1,-1), y_score=ypred.reshape(1,-1), k=ordered_idxs.shape[0])

def test_score_sanity():
    ytrue = np.zeros(10000)
    randorder = np.random.permutation(10000)
    
    numpos = 100
    posidxs = randorder[:numpos]
    negidxs = randorder[numpos:]
    numneg = negidxs.shape[0]

    ytrue[posidxs] = 1.
    perfect_rank = np.argsort(-ytrue)
    bad_rank = np.argsort(ytrue)

    ## check score for a perfect result set is  1
    ## regardless of whether the rank is computed when k < numpos or k >> numpos
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos//2]),1.)
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos]),1.)
    assert np.isclose(ndcg_rank_score(ytrue,perfect_rank[:numpos*2]),1.)    
    
        ## check score for no results is 0    
    assert np.isclose(ndcg_rank_score(ytrue,bad_rank[:-numpos]),0.)

  
    ## check score for same amount of results worsens if they are shifted    
    gr = perfect_rank[:numpos//2]
    br = bad_rank[:numpos//2]
    rank1 = np.concatenate([gr,br])
    rank2 = np.concatenate([br,gr])
    assert ndcg_rank_score(ytrue, rank1) > .5, 'half of entries being relevant, but first half'
    assert ndcg_rank_score(ytrue, rank2) < .5

def test_score_rare():
    n = 10000 # num items
    randorder = np.random.permutation(n)

    ## check in a case with only few positives
    for numpos in [0,1,2]:
        ytrue = np.zeros_like(randorder)
        posidxs = randorder[:numpos]
        negidxs = randorder[numpos:]
        numneg = negidxs.shape[0]
    
        ytrue[posidxs] = 1.
        perfect_rank = np.argsort(-ytrue)
        bad_rank = np.argsort(ytrue)
    
        scores = []
        k = 200
        for i in range(k):
            test_rank = bad_rank[:k].copy()
            if len(posidxs) > 0:
                test_rank[i] = posidxs[0]
            sc = ndcg_rank_score(ytrue,test_rank)
            scores.append(sc)
        scores = np.array(scores)
        if numpos == 0:
            assert np.isclose(scores ,0).all()
        else:
            assert (scores > 0 ).all()

test_score_sanity()
test_score_rare()


def compute_metrics(results, indices, gt):
    hits = results
    assert ~gt.isna().any()
    ndcg_score = ndcg_rank_score(gt.values, indices)    
    assert (gt.iloc[indices].values.astype('float') == hits.astype('float')).all()
    
    hpos= np.where(hits > 0)[0] # > 0 bc. nan.
    if hpos.shape[0] > 0:
        nfirst = hpos.min() + 1
        rr = 1./nfirst
    else:
        nfirst = np.inf
        rr = 1/nfirst
        
    nfound = (hits > 0).cumsum()
    ntotal = (gt > 0).sum()
    nframes = np.arange(hits.shape[0]) + 1
    precision = nfound/nframes
    recall = nfound/ntotal
    total_prec = (precision*hits).cumsum() # see Evaluation of ranked retrieval results page 158
    average_precision = total_prec/ntotal
    
    return dict(ndcg_score=ndcg_score, ntotal=ntotal, nfound=nfound[-1], 
                ndatabase=gt.shape[0], abundance=(gt > 0).sum()/gt.shape[0], nframes=hits.shape[0],
                nfirst = nfirst, reciprocal_rank=rr,
               precision=precision[-1], recall=recall[-1], average_precision=average_precision[-1])


def side_by_side_comparison(stats, baseline_variant, metric):
    v1 = stats
    metrics = list(set(['nfound', 'nfirst'] + [metric]))

    v1 = v1[['dataset', 'category', 'variant', 'ntotal', 'abundance'] + metrics]
    v2 = stats[stats.variant == baseline_variant]
    rename_dict = {}
    for m in metrics:
        rename_dict[metric] = f'base_{metric}'
        
    rename_dict['base_variant'] = baseline_variant
    
    v2 = v2[['dataset', 'category'] + metrics]
    v2['base'] = v2[metric]
    v2 = v2.rename(mapper=rename_dict, axis=1)
    
    sbs = v1.merge(v2, right_on=['dataset', 'category'], left_on=['dataset', 'category'])
    sbs = sbs.assign(ratio=sbs[metric]/sbs['base'])
    sbs = sbs.assign(delta=sbs[metric] - sbs['base'])
    return sbs


import datetime
import pickle
def dump_results(benchresults):
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y-%m-%d_%H:%M:%S")
    fname = './data/vls_bench_{}.pkl'.format(nowstr)
    pickle.dump(benchresults, open(fname,'wb'))
    print(fname)