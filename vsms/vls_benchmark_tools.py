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

# ignore this comment

def vls_init_logger():
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)

def brief_formatter(num_sd):
    assert num_sd > 0
    def formatter(ftpt):
        if math.isclose(ftpt, 0.):
            return '0'
        
        if math.isclose(ftpt,1.):
            return '1'

        if ftpt < 1.:
            exp = -math.floor(math.log10(abs(ftpt)))
            fmt_string = '{:.0%df}' % (exp + num_sd - 1)
            dec = fmt_string.format(ftpt)    
        else:
            fmt_string = '{:.02f}'
            dec = fmt_string.format(ftpt)
            
        zstripped = dec.lstrip('0').rstrip('0')
        return zstripped.rstrip('.')
    return formatter

brief_format = brief_formatter(1)

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

    offset = left_excess - right_excess

    x1p = x1 + offset
    x2p = x2 + offset
    x1p = np.clip(x1p,0,max_x) # trim any approx error
    x2p = np.clip(x2p,0,max_x) # same

    assert np.isclose(x2p - x1p, x2 - x1).all()
    assert (x1p >= 0).all(), x1p
    assert (x2p <= max_x).all(), x2p
    return x1p, x2p

def random_seg_start(x1, x2, target_x, max_x, off_center_range, n=1):
    dist = x2 - x1
    # assert (dist <= target_x).all(), dont enforce containment
    center = (x2 + x1)/2

    rel_offset = (np.random.rand(n) - .5)*off_center_range

    ## perturb offset a bit. but do keep center within crop
    if (dist < target_x).all():
        offset = rel_offset * (target_x - dist) * .5
    else:
        assert (dist >= target_x).all() # figure out what to do when some are and some arent.
        offset = rel_offset * target_x * .5
    
    start = center - target_x*.5
    start = start + offset
    end = start + target_x
    start, end = readjust_interval(start, end, max_x)
    assert np.isclose(end - start, target_x).all()
    assert (start <= center).all()
    assert (center <= end).all()
    return start, end

def add_clearance(x1,x2,max_x, clearance_ratio):
    cx = (x1 + x2)*.5
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
    max_len = min(b.im_height, b.im_width)
    img_scale = max_len/max_d

    min_scale = min(img_scale, clearance)
    max_scale = min(img_scale, scale_range*clearance) # don't do more than 3x
    assert img_scale >= max_scale >= min_scale

    scale = np.exp(np.random.rand(n)*np.log(max_scale/min_scale))*min_scale
    # assert (scale >= clearance).all()
    # assert (scale <= scale_range).all()

    target_x = scale*max_d
    target_y = target_x
    assert ((bw <= target_x) | (target_x == max_len)).all()
    assert ((bh <= target_y) | (target_y == max_len)).all()
    
    if False:
        lratio = 2*(np.random.rand(n) - .5)*np.log(aspect_ratio_range)
        ratio = np.exp(lratio/2)

        upper = math.sqrt(aspect_ratio_range)
        assert (ratio <= upper).all()
        assert (ratio >= 1/upper).all()

        ## TODO: after adjusting the box, it is possible that we violate prevously valid constraints wrt. 
        ## the object box or wrt the containing image. The ratio limits need to be bound based on these constraints
        ## before applying randomness
    else:
        ratio = 1.


    target_y = target_y*ratio
    target_x = target_x/ratio #np.ones_like(ratio)
    start_x, end_x = random_seg_start(b.x1, b.x2, target_x, b.im_width, off_center_range=off_center_range, n=n)
    start_y, end_y = random_seg_start(b.y1, b.y2, target_y, b.im_height, off_center_range=off_center_range, n=n)
    
    assert ((bw > target_x) | (start_x <= b.x1)).all()
    assert ((bw > target_x) | (end_x >= b.x2)).all()

    return pd.DataFrame({'x1':start_x, 'x2': end_x, 'y1':start_y, 'y2':end_y})

def randomly_extended_crop(im, box, scale_range, aspect_ratio_range, off_center_range, clearance, n):
    rbs = random_container_box(box, scale_range, aspect_ratio_range, off_center_range, clearance, n=n)
    crs = []
    for cb in rbs.itertuples():
        cr = im.crop((cb.x1, cb.y1, cb.x2, cb.y2))
        crs.append(cr)
    return crs

def process_crops(crs, tx, embedding):
    if len(crs) == 0:
        return np.zeros((0,512))

    tensors = []
    for cr in crs:
        cr = cr.resize((224,224), resample=3)
        ts = tx(cr)
        tensors.append(ts)

    allts = torch.stack(tensors)

    emvecs = []
    bs = 20
    for i in range(0,len(allts), bs):
        batch = allts[i:i+bs]
        embs = embedding.from_image(preprocessed_image=batch, pooled='bypass')
        emvecs.append(embs)

    embs = np.concatenate(emvecs)
    return embs


def run_loop6(*, ev :EvDataset, category, qstr, interactive, warm_start, n_batches, batch_size, minibatch_size, 
              learning_rate, max_examples, num_epochs, loss_margin, 
              tqdm_disabled:bool, granularity:str,
               positive_vector_type, n_augment,
               model_type='logistic', solver_opts={}, 
               **kwargs):         
    assert 'fine_grained' not in kwargs
    assert isinstance(granularity, str)
    assert positive_vector_type in ['image_only', 'image_and_vec', 'vec_only', None]
    # gvec = ev.fine_grained_embedding
    # gvec_meta = ev.fine_grained_meta
    # min_box_size = 60 # in pixels. fov is 224 min. TODO: ground truth should ignore these.
    min_box_size = 10
    augment_n = n_augment # number of random augments 
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
        vec_meta = ev.fine_grained_meta
        vecs = ev.fine_grained_embedding
        #index_path = './data/bdd_10k_allgrains_index.ann'
        index_path = None
        hdb = AugmentedDB(raw_dataset=ev.image_dataset, embedding=ev.embedding, 
            embedded_dataset=vecs, vector_meta=vec_meta, index_path=index_path)

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
    acc_vecs = [np.zeros((0,512))]
            
    gt = ev.query_ground_truth[category].values
    res = {'indices':acc_indices, 'results':acc_results}#, 'gt':gt.values.copy()}
    if init_vec is not None:
        init_vec = init_vec/np.linalg.norm(init_vec)
        tvec = init_vec#/np.linalg.norm(init_vec)
    else:
        tvec = None

    tmode = init_mode
    clip_tx = T.Compose([
                    T.ToTensor(), 
                    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    lambda x : x.type(torch.float16)
                    ])

    for i in tqdm(range(n_batches), leave=False, disable=tqdm_disabled):
        idxbatch = bfq.query_stateful(mode=tmode, vector=tvec, batch_size=batch_size)
        acc_indices.append(idxbatch.copy()) # trying to figure out leak
        acc_results.append(gt[idxbatch])

        if interactive != 'plain':
            if granularity in ['fine', 'multi']:
                batchpos, batchneg = get_pos_negs_all_v2(idxbatch, ds, vec_meta)
                ## we are currently ignoring these positives
                acc_pos.append(batchpos)
                acc_neg.append(batchneg)

                pos = pr.BitMap.union(*acc_pos)
                neg = pr.BitMap.union(*acc_neg)

                if positive_vector_type in ['image_only', 'image_and_vec']:
                    crs = []
                    for idx in idxbatch:
                        boxes = ds[idx]
                        widths = (boxes.x2 - boxes.x1)
                        heights = (boxes.y2 - boxes.y1)
                        boxes = boxes[(widths >= min_box_size) & (heights >= min_box_size)]

                        if boxes.shape[0] == 0:
                            continue
                        # only read image if there is something
                        im = imds[idx]

                        for b in boxes.itertuples():
                            if augment_n > 1:
                                pcrs = randomly_extended_crop(im, b, scale_range=3., aspect_ratio_range=1., off_center_range=1., 
                                        clearance=1.3, n=augment_n)
                                for cr in pcrs:
                                    cr = T.RandomHorizontalFlip()(cr)
                                    crs.append(cr)
                            else:
                                pcrs = randomly_extended_crop(im, b, scale_range=1., aspect_ratio_range=1., off_center_range=0., 
                                        clearance=1.5, n=1)

                                for cr in pcrs:
                                    crs.append(cr)
                
                    tmp = process_crops(crs, clip_tx, ev.embedding)
                    acc_vecs.append(tmp)
                    impos = np.concatenate(acc_vecs)

                if positive_vector_type == 'image_only':
                    allpos = impos  
                if positive_vector_type == 'vec_only':
                    allpos = vecs[pos]
                elif positive_vector_type == 'image_and_vec':
                    allpos = np.concatenate([impos, vecs[pos]])

                Xt = np.concatenate([allpos, vecs[neg]])
                yt = np.concatenate([np.ones(len(allpos)), np.zeros(len(neg))])

                # not really valid. some boxes are area 0. they should be ignored.but they affect qgt
                # if np.concatenate(acc_results).sum() > 0:
                #    assert len(pos) > 0
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



def process_tups(evs, benchresults, keys, at_N):
    tups = []
    ## lvis: qgt has 0,1 and nan. 0 are confirmed negative. 1 are confirmed positive.
    for k in keys:#['lvis','dota','objectnet', 'coco', 'bdd' ]:#,'bdd','ava', 'coco']:
        val = benchresults[k]
        ev = evs[k]
    #     qgt = benchgt[k]
        for tup,exp in val:
            hits = np.concatenate(exp['results'])
            indices = np.concatenate(exp['indices'])
            index_set = pr.BitMap(indices)
            assert len(index_set) == indices.shape[0], 'check no repeated indices'
            #ranks = np.concatenate(exp['ranks'])
            #non_nan = ~np.isnan(ranks)
            gtfull = ev.query_ground_truth[tup['category']].clip(0,1)
            gt = gtfull[~gtfull.isna()]
            idx_map = dict(zip(gt.index,np.arange(gt.shape[0]).astype('int')))
            local_idx = np.array([idx_map[idx] for idx in indices])
            metrics = compute_metrics(hits[:at_N], local_idx[:at_N], gt)
    #         if tup['category'] == 'snowy weather':
    #             break

            output_tup = {**tup, **metrics}
            #output_tup['unique_ranks'] = np.unique(ranks[non_nan]).shape[0]
            output_tup['dataset'] = k
            tups.append(output_tup)
            
    return pd.DataFrame(tups)

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

def better_same_worse(stats, variant, baseline_variant='plain', metric='ndcg_score', reltol=1.1,
                     summary=True):
    sbs = side_by_side_comparison(stats, baseline_variant='plain', metric='ndcg_score')
    invtol = 1/reltol
    sbs = sbs.assign(better=sbs.ndcg_score > reltol*sbs.base, worse=sbs.ndcg_score < invtol*sbs.base,
                    same=sbs.ndcg_score.between(invtol*sbs.base, reltol*sbs.base))
    bsw = sbs[sbs.variant == variant].groupby('dataset')[['better', 'same', 'worse']].sum()
    if summary:
        return bsw
    else:
        return sbs


import datetime
import pickle
def dump_results(benchresults):
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y-%m-%d_%H:%M:%S")
    fname = './data/vls_bench_{}.pkl'.format(nowstr)
    pickle.dump(benchresults, open(fname,'wb'))
    print(fname)