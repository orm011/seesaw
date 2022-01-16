#from .data_server import *
from .search_loop_models import *
from .search_loop_tools import *

import inspect
from .dataset_tools import *
from .vloop_dataset_loaders import EvDataset, get_class_ev
from .fine_grained_embedding import *
from .multiscale_index import *
from .coarse_index import CoarseIndex
from .search_loop_models import adjust_vec, adjust_vec2
import numpy as np
import sklearn.metrics
import math
from .util import *
from .pairwise_rank_loss import VecState
import pyroaring as pr
import importlib
from .dataset_manager import VectorIndex
from .seesaw_session import SeesawLoop, LoopParams

from .figures import ndcg_score_fn

# ignore this comment

def vls_init_logger():
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)

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

import time

_clip_tx = T.Compose([
                    T.ToTensor(), 
                    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    lambda x : x.type(torch.float16)
                    ])


import random
from .figures import compute_metrics
from .server_session_state import Session

def benchmark_loop(*, hdb0 : AccessMethod, n_batches, tqdm_disabled:bool, category, qstr,
                interactive, warm_start, batch_size, minibatch_size, 
              learning_rate, max_examples, num_epochs, loss_margin, 
              max_feedback=None, box_drop_prob=0.,
               granularity:str, positive_vector_type, n_augment,min_box_size=10,
               model_type='logistic', solver_opts={}, **kwargs):     
    assert positive_vector_type in ['image_only', 'image_and_vec', 'vec_only', None]
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    params = {k:v for (k,v) in values.items() if k in args and k not in ['ev', 'category', 'qstr', 'n_batches', 'max_feedback', 'box_drop_prob']} 
    rtup = {**{'category':category, 'qstr':qstr}, **params, **kwargs}       

    catgt = ev0.query_ground_truth[category]
    class_idxs = catgt[~catgt.isna()].index.values
    if class_idxs.shape[0] == 0:
        print(f'No labelled frames for class "{category}" found ')
        return (rtup, None)

    ev, class_idxs = get_class_ev(ev0, category, boxes=True)
    ds =  DataFrameDataset(ev.box_data[ev.box_data.category == category], index_var='dbidx', max_idx=class_idxs.shape[0]-1)
    gt0 = ev.query_ground_truth[category]
    gt = gt0.values

    rtup['nbatches'] = n_batches
    rtup['ntotal'] = gt.sum()
    rtup['nimages'] = gt.shape[0]
    rtup['nvecs'] = ev.fine_grained_meta.shape[0]

    print('benchmark loop', rtup)
    params = LoopParams(**params)

    max_results = gt.sum()
    assert max_results > 0
    
    acc_indices = []
    acc_results = []

    total_results = 0
    loop = SeesawLoop(ev, params)
    loop.set_vec(qstr=qstr)
    start_time = time.time()
    images_seen = pr.BitMap()

    for i in tqdm(range(n_batches),leave=False, disable=tqdm_disabled):
        print(f'iter {i}')
        if i >= 1:
            curr_time = time.time()
            print(f'previous iteration took {curr_time - start_time}s')
            start_time = time.time()

        idxbatch = loop.next_batch()
        
        # ran out of batches
        if idxbatch.shape[0] == 0:
            break

        #images_seen.update(idxbatch)
        acc_indices.append(idxbatch)
        acc_results.append(gt[idxbatch])
        total_results += gt[idxbatch].sum()

        if total_results == max_results:
            print(f'Found all {total_results} possible results for {category} after {i} batches. stopping...')
            break

        if i + 1 == n_batches:
            print(f'n batches. ending...')
            break 
        
        box_dict = {}
        for idx in idxbatch:
            bxs = ds[idx]
            rnd = np.random.rand(bxs.shape[0])
            bxs = bxs[rnd >= box_drop_prob]
            box_dict[idx] = bxs

        gt2 = np.array([box_dict[idx].shape[0] > 0 for idx in idxbatch]).astype('float')
        # if (gt2 != gt[idxbatch]).all():
        #     print('Warning: gt data and box data seem to disagree. ')

        if max_feedback is None or (i+1)*batch_size <= max_feedback:
            loop.refine(idxbatch=idxbatch, box_dict=box_dict)

    res = {}
    hits = np.concatenate(acc_results)
    indices = np.concatenate(acc_indices)

    assert hits.shape[0] == indices.shape[0]
    index_set = pr.BitMap(indices)
    assert len(index_set) == indices.shape[0], 'check no repeated indices'
    res['hits'] = np.where(hits)[0]
    res['total_seen'] = indices.shape[0]
    res['latency_profile'] = pd.DataFrame.from_records(loop.state.latency_profile)

    return (rtup, res)

def run_on_actor(br, tup):
    return br.run_loop.remote(tup)

from .progress_bar import tqdm_map

#from .dataset_manager import RemoteVectorIndex
import os

class BenchRunner(object):
    def __init__(self, evs):
        print('initing benchrunner env...')
        print(evs)
        revs = {}
        for (k,evref) in evs.items():
            if isinstance(evref, ray.ObjectRef): # remote
                ev = ray.get(evref)
                revs[k] = ev
            else: # local
                revs[k] = evref
        self.evs = revs

        assert os.path.exists(vector_path), vector_path
        vi = VectorIndex(load_path=vector_path, copy_to_tmpdir=True, prefault=True)
        self.evs[k].vec_index = vi # use vector store directly instead

        vls_init_logger()
        print('loaded all evs...')

    def ready(self):
        return True
    
    def run_loop(self, tup):
        import seesaw
        importlib.reload(seesaw)

        start = time.time()
        print(f'getting ev for {tup["dataset"]}...')
        ev = self.evs[tup['dataset']]
        print(f'got ev {ev} after {time.time() - start}')
        ## for repeated benchmarking we want to reload without having to recreate these classes
        # importlib.reload(importlib.import_module('seesaw'))
        # importlib.reload(importlib.import_module(benchmark_loop.__module__))
        res = seesaw.benchmark_loop(ev=ev, **tup)
        print(f'Finished running after {time.time() - start}')
        return res

RemoteBenchRunner = ray.remote(BenchRunner)

def make_bench_actors(evs, num_actors, resources=dict(num_cpus=4, memory=15*(2**30))):
    evref = ray.put(evs)
    actors = []
    try:
        for i in range(num_actors):
            a = RemoteBenchRunner.options(**resources).remote(evs=evref)
            actors.append(a)
    except Exception as e:
        for a in actors:
            ray.kill(a)
        raise e

    return actors

def parallel_run(*, evs, actors, tups, benchresults):
    print('new run')
    if len(actors) > 0:
        tqdm_map(actors, run_on_actor, tups, benchresults)
    else:    
        for tup in tqdm(tups):
            pexp = BenchRunner(evs).run_loop(tup)
            benchresults.append(pexp)