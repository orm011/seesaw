from dataclasses import dataclass
import random
import copy
from seesaw.figures import compute_metrics


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
from .basic_types import *

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



def fill_imdata(imdata : Imdata, box_data : pd.DataFrame, b : BenchParams):
    imdata = imdata.copy()
    rows = box_data[box_data.dbidx == imdata.dbidx]
    if rows.shape[0] > 0:
      rows = rows.assign(description=rows.category, marked_accepted=(rows.category == b.ground_truth_category))
      rows = rows[['x1', 'x2', 'y1', 'y2', 'description', 'marked_accepted']]

      ## drop some boxes based on b.box_drop_prob 
      rnd = np.random.rand(rows.shape[0])
      kept_rows = rows[rnd >=  b.box_drop_prob]
      filling = [Box(**b) for b in kept_rows.to_dict(orient='records')]
    else:
      filling = []
    imdata.boxes = filling
    imdata.marked_accepted = is_accepted(imdata)
    return imdata

def benchmark_loop(*, session : Session,  subset : pr.FrozenBitMap, box_data : pd.DataFrame,
                      b : BenchParams, p : SessionParams):

    all_box_data = box_data
    box_data = box_data[box_data.category==b.ground_truth_category]
    positives = pr.FrozenBitMap(box_data.dbidx.values)

    assert positives.intersection(subset) == positives, 'index mismatch'
    max_results = min(len(positives),b.max_results)
    assert max_results > 0

    total_results = 0
    total_seen = 0
    session.set_text(b.qstr)
    for batch_num in tqdm(range(1, b.n_batches + 1),leave=False, disable=True):
        print(f'iter {batch_num}')
        idxbatch = session.next()
        assert [idx in subset for idx in idxbatch]
        if len(idxbatch) == 0:
            break

        s = copy.deepcopy(session.get_state())
        last_batch = s.gdata[-1]
        for j, imdata in enumerate(last_batch):
          if p.interactive == 'textual':
            last_batch[j] = fill_imdata(imdata, all_box_data, b)
          else:            
            last_batch[j] = fill_imdata(imdata, box_data, b)

        session.update_state(s)
        batch_pos = np.array([imdata.marked_accepted for imdata in last_batch])
        total_results += batch_pos.sum()
        total_seen += idxbatch.shape[0]

        if total_results == max_results:
            print(f'Found all {total_results} possible results for {b.ground_truth_category} after {batch_num} batches. stopping...')
            break

        if batch_num == b.n_batches:
            print(f'iter {batch_num} = {b.n_batches}. ending...')
            break 
        
        if b.max_feedback is None or (batch_num+1)*p.batch_size <= b.max_feedback:
            print('refining...')
            session.refine()

    return dict(nfound=int(total_results), nseen=int(total_seen))

def run_on_actor(br, tup):
    return br.run_loop.remote(*tup)

from .progress_bar import tqdm_map

import os
import string
import time


class BenchRunner(object):
    def __init__(self, seesaw_root, results_dir):
        assert os.path.isdir(results_dir)
        vls_init_logger()
        self.gdm = GlobalDataManager(seesaw_root)
        self.results_dir = results_dir
        random.seed(int(f'{time.time_ns()}{os.getpid()}'))
        
    def ready(self):
        return True

    def run_loop(self, b : BenchParams, p : SessionParams):
        import seesaw
        importlib.reload(seesaw)
        from seesaw import prep_bench_data, benchmark_loop
        start = time.time()

        random_suffix = ''.join([random.choice(string.ascii_lowercase) for _ in range(10)])
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f'{self.results_dir}/session_{timestamp}_{random_suffix}'
        os.mkdir(output_dir)
        summary = BenchSummary(bench_params=b, session_params=p, 
          timestamp = timestamp, result=None)

        output_path = f'{output_dir}/summary.json'
        json.dump(summary.dict(),open(output_path, 'w'))
        print(f'saving output to {output_dir}')

        assert p.index_spec.c_name is not None, 'need category for benchmark'

        try:
          ds = self.gdm.get_dataset(p.index_spec.d_name)
          hdb = self.gdm.load_index(p.index_spec.d_name, p.index_spec.i_name)

          box_data, subset, positive  = prep_bench_data(ds, p)
          
          if len(positive) == 0:
            print('no frames available, exiting')
            return output_dir 

          hdb = hdb.subset(subset)

          session = Session(self.gdm, ds, hdb, p)          
          run_info = benchmark_loop(session=session, box_data=box_data, subset=subset, b=b, p=p)
          summary.result = BenchResult(ntotal=len(positive), nimages = len(subset), 
                                      session=session.get_state(), run_info=run_info, total_time=time.time() - start)
        finally:
          json.dump(summary.dict(), open(output_path, 'w'))
        
        return output_dir

import glob

def get_metric_summary(session : SessionState):
    curr_idx = 0
    hit_indices = []
    for ent in session.gdata:
        for imdata in ent:
            if imdata.marked_accepted:
                hit_indices.append(curr_idx)
            curr_idx +=1
    index_set = pr.BitMap(hit_indices)
    assert len(index_set) == len(hit_indices)
    return dict(hit_indices=np.array(index_set), total_seen=curr_idx)

def process_one_row(obj, path, at_N):
    base_path = path[:-len('summary.json')]
    bs = BenchSummary(**json.load(open(path)))
    b = bs.bench_params
    s = bs.session_params

    res = {**b.dict(), **s.index_spec.dict(), **s.dict()}
    res['session_path'] = base_path

    if bs.result is not None:
        summary = get_metric_summary(bs.result.session)
        mets = compute_metrics(**summary, batch_size=s.batch_size, total_positives=bs.result.ntotal, ndatabase=bs.result.nimages, at_N=at_N)
        res.update(**mets)

    return res

def get_metrics_table(base_path, at_N):
    summary_paths = glob.glob(base_path + '/**/summary.json')
    res = []

    r = ray.data.read_binary_files(summary_paths)
    mp = r.map(lambda b : json.loads(b.decode()))

    acc = []
    for obj,path in zip(mp.iter_rows(), summary_paths):
      res = process_one_row(obj, path, at_N)
      acc.append(res)

    return pd.DataFrame(acc)

def prep_bench_data(ds, p : SessionParams):
    box_data, qgt = ds.load_ground_truth()
    catgt = qgt[p.index_spec.c_name]    
    
    positive_box_data = box_data[box_data.category == p.index_spec.c_name]
    present = pr.FrozenBitMap(catgt[~catgt.isna()].index.values)
    positive = pr.FrozenBitMap(positive_box_data.dbidx.values)

    assert positive.intersection(present) == positive    
    return box_data, present, positive

from .dataset_manager import IndexSpec
from .dataset_search_terms import category2query
def gen_configs(gdm : GlobalDataManager, datasets, variants, s_template : SessionParams, b_template : BenchParams, 
      max_classes_per_dataset=math.inf):
    configs = []
    avail_datasets = gdm.list_datasets()
    for d in datasets:
        assert d in avail_datasets
        ds = gdm.get_dataset(d)
        _,qgt=ds.load_ground_truth()
        ctpos = qgt.sum()
        classes = ctpos.index[(ctpos > 0)]
        for i,c in enumerate(classes):
            if i > max_classes_per_dataset:
                break
            for var in variants:
                update_b = {}
                update_b['qstr'] = category2query(d.split('/')[-2], c)
                update_b['ground_truth_category'] = c
                b = BenchParams(**{**b_template, 
                                   **update_b, 
                                   **{k:v for (k,v) in var.items() if k in BenchParams.__fields__.keys()}}
                               )
                
                update_s = {'index_spec':IndexSpec(d_name=d, i_name=var['index_name'], c_name=c)}
                s = SessionParams(**{**s_template,
                                     **update_s,
                                     **{k:v for (k,v) in var.items() if k in SessionParams.__fields__.keys()}}
                                 )
                configs.append((b,s))
    return configs

RemoteBenchRunner = ray.remote(BenchRunner)

def make_bench_actors(resources_per_bench, bench_constructor_args):
    #dict(num_cpus=4, memory=15*(2**30))
    resources = resources_per_bench

    avail_res = ray.available_resources()
    num_nodes = len(ray.nodes())

    max_mem = math.floor(avail_res['memory']//num_nodes//resources['memory'])
    max_cpu = math.floor(avail_res['CPU']//num_nodes//resources['num_cpus'])

    num_actors_per_node = min(max_mem, max_cpu) - 2
    num_actors = num_actors_per_node * num_nodes
    print(f'creating {num_actors} based on available shares: mem {max_mem} cpu {max_cpu}')
    actors = []
    try:
        for i in range(num_actors):
            a = RemoteBenchRunner.options(**resources).remote(**bench_constructor_args)
            actors.append(a)
    except Exception as e:
        for a in actors:
            ray.kill(a)
        raise e 

    return actors

def parallel_run(*, actors, tups):
  res  = []
  tqdm_map(actors, run_on_actor, tups, res=res)