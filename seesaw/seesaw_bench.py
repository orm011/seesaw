from dataclasses import dataclass
import random
import copy


from .search_loop_models import *
from .search_loop_tools import *

import inspect
from .dataset_tools import *
from .fine_grained_embedding import *
from .multiscale_index import *
from .coarse_index import CoarseIndex
from .search_loop_models import adjust_vec, adjust_vec2
import numpy as np
import math
from .util import *
import pyroaring as pr
import importlib
from .seesaw_session import SessionParams, Session, Imdata, Box

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



@dataclass(frozen=True)
class BenchParams:
    ground_truth_category : str
    dataset_name : str
    index_name : str
    qstr : str
    n_batches : int
    max_feedback : int
    box_drop_prob : float


def fill_imdata(imdata : Imdata, box_data : pd.DataFrame, b : BenchParams):
    imdata = imdata.copy()
    rows = box_data[box_data.dbidx == imdata.dbidx]
    if rows.shape[0] > 0:
      rows = rows[['x1', 'x2', 'y1', 'y2', 'category']]

      ## drop some boxes based on b.box_drop_prob 
      rnd = np.random.rand(rows.shape[0])
      kept_rows = rows[rnd >=  b.box_drop_prob]
      filling = [Box(**b) for b in kept_rows.to_dict(orient='records')]
    else:
      filling = []
    imdata.boxes = filling
    imdata.marked_accepted = len(filling) > 0
    return imdata

def benchmark_loop(*, session : Session,  subset : pr.FrozenBitMap, box_data : pd.DataFrame,
                      b : BenchParams, p : SessionParams):
    rtup = {**p.__dict__, **b.__dict__}    
    positives = pr.FrozenBitMap(box_data.dbidx.values)
    assert positives.intersection(subset) == positives, 'index mismatch'
    max_results = len(positives)
    assert max_results > 0
    rtup['ntotal'] = max_results
    rtup['nimages'] = len(subset)

    acc_results = []
    print('benchmark loop', rtup)

    total_results = 0
    session.set_text(b.qstr)

    for i in tqdm(range(b.n_batches),leave=False, disable=True):
        print(f'iter {i}')
        idxbatch = session.next()
        assert [idx in subset for idx in idxbatch]
        if len(idxbatch) == 0:
            break

        s = copy.deepcopy(session.get_state())
        last_batch = s.gdata[-1]
        for i, imdata in enumerate(last_batch):
          last_batch[i] = fill_imdata(imdata, box_data, b)

        session.update_state(s)
        
        batch_pos = np.array([imdata.marked_accepted for imdata in last_batch])
        acc_results.append(batch_pos)
        total_results += batch_pos.sum()

        if total_results == max_results:
            print(f'Found all {total_results} possible results for {b.ground_truth_category} after {i} batches. stopping...')
            break

        if i + 1 == b.n_batches:
            print(f'n batches. ending...')
            break 
        
        if b.max_feedback is None or (i+1)*p.batch_size <= b.max_feedback:
            session.refine()

    res = {}
    hits = np.concatenate(acc_results)
    indices = np.concatenate(session.acc_indices)
    assert hits.shape[0] == indices.shape[0]
    index_set = pr.BitMap(indices)
    assert len(index_set) == indices.shape[0], 'check no repeated indices'
    res['hits'] = np.where(hits)[0]
    res['total_seen'] = indices.shape[0]
    res['latency_profile'] = np.array(session.timing)

    return (rtup, res)

def run_on_actor(br, tup):
    return br.run_loop.remote(tup)

from .progress_bar import tqdm_map

import os
import string

class BenchRunner(object):
    def __init__(self, seesaw_root, results_dir):
        assert os.path.isdir(results_dir)
        vls_init_logger()
        self.gdm = GlobalDataManager(seesaw_root)
        self.results_dir = results_dir
        random.seed(os.getpid())
        
    def ready(self):
        return True

    def run_loop(self, b : BenchParams, p : SessionParams):
        import seesaw
        importlib.reload(seesaw)
        from seesaw import Session, prep_bench_data, benchmark_loop
        start = time.time()

        session, box_data, subset_idxs = prep_bench_data(self.gdm, b, p)
        random_suffix = ''.join([random.choice(string.ascii_lowercase) for _ in range(10)])
        output_dir = f'{self.results_dir}/session_{time.strftime("%Y%m%d-%H%M%S")}_{random_suffix}'
        os.mkdir(output_dir)
        print(f'saving results to {output_dir}')
        res = benchmark_loop(session=session, box_data=box_data, subset=subset_idxs, b=b, p=p)
        json.dump(b.__dict__,open(f'{output_dir}/bench_params.json', 'w'))
        session.save_state(f'{output_dir}/session_state.json')

        print(f'Finished running after {time.time() - start}')
        return res

def prep_bench_data(gdm : GlobalDataManager,  b : BenchParams, p : SessionParams):
    ds = gdm.get_dataset(b.dataset_name)
    box_data, qgt = ds.load_ground_truth()

    catgt = qgt[b.ground_truth_category]    
    box_data = box_data[box_data.category == b.ground_truth_category]
    subset_idxs = pr.FrozenBitMap(catgt[~catgt.isna()].index.values)

    if len(subset_idxs) == 0:
        print(f'No labelled frames for class "{b.ground_truth_category}" found ')
        return None
    
    hdb = gdm.load_index(b.dataset_name, b.index_name)
    hdb = hdb.subset(subset_idxs)
    session = Session(gdm, ds, hdb, p)

    return session, box_data, subset_idxs

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