import random
import copy

from .search_loop_models import *
from .search_loop_tools import *

from .dataset_tools import *
from .fine_grained_embedding import *

# from .multiscale.multiscale_index import *
from .search_loop_models import adjust_vec, adjust_vec2
import numpy as np
import math
from .util import *
import pyroaring as pr
from .seesaw_session import SessionParams, Session, Imdata, Box, make_session
from .basic_types import *
from .metrics import compute_metrics
from .dataset_manager import GlobalDataManager
import numpy as np


def readjust_interval(x1, x2, max_x):
    left_excess = -np.clip(x1, -np.inf, 0)
    right_excess = np.clip(x2 - max_x, 0, np.inf)

    offset = left_excess - right_excess

    x1p = x1 + offset
    x2p = x2 + offset
    x1p = np.clip(x1p, 0, max_x)  # trim any approx error
    x2p = np.clip(x2p, 0, max_x)  # same

    assert np.isclose(x2p - x1p, x2 - x1).all()
    assert (x1p >= 0).all(), x1p
    assert (x2p <= max_x).all(), x2p
    return x1p, x2p


def random_seg_start(x1, x2, target_x, max_x, off_center_range, n=1):
    dist = x2 - x1
    # assert (dist <= target_x).all(), dont enforce containment
    center = (x2 + x1) / 2

    rel_offset = (np.random.rand(n) - 0.5) * off_center_range

    ## perturb offset a bit. but do keep center within crop
    if (dist < target_x).all():
        offset = rel_offset * (target_x - dist) * 0.5
    else:
        assert (
            dist >= target_x
        ).all()  # figure out what to do when some are and some arent.
        offset = rel_offset * target_x * 0.5

    start = center - target_x * 0.5
    start = start + offset
    end = start + target_x
    start, end = readjust_interval(start, end, max_x)
    assert np.isclose(end - start, target_x).all()
    assert (start <= center).all()
    assert (center <= end).all()
    return start, end


def add_clearance(x1, x2, max_x, clearance_ratio):
    cx = (x1 + x2) * 0.5
    dx = x2 - x1
    diff = dx * clearance_ratio * 0.5
    return readjust_interval(cx - diff, cx + diff, max_x)


def add_box_clearance(b, max_x, max_y, clearance_ratio):
    x1, x2 = add_clearance(b.x1, b.x2, max_x, clearance_ratio)
    y1, y2 = add_clearance(b.y1, b.y2, max_y, clearance_ratio)
    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}


def random_container_box(
    b, scale_range=3.3, aspect_ratio_range=1.2, off_center_range=1.0, clearance=1.2, n=1
):
    assert clearance >= aspect_ratio_range

    bw = b.x2 - b.x1
    bh = b.y2 - b.y1
    max_d = max(bw, bh)
    max_len = min(b.im_height, b.im_width)
    img_scale = max_len / max_d

    min_scale = min(img_scale, clearance)
    max_scale = min(img_scale, scale_range * clearance)  # don't do more than 3x
    assert img_scale >= max_scale >= min_scale

    scale = np.exp(np.random.rand(n) * np.log(max_scale / min_scale)) * min_scale
    # assert (scale >= clearance).all()
    # assert (scale <= scale_range).all()

    target_x = scale * max_d
    target_y = target_x
    assert ((bw <= target_x) | (target_x == max_len)).all()
    assert ((bh <= target_y) | (target_y == max_len)).all()

    if False:
        lratio = 2 * (np.random.rand(n) - 0.5) * np.log(aspect_ratio_range)
        ratio = np.exp(lratio / 2)

        upper = math.sqrt(aspect_ratio_range)
        assert (ratio <= upper).all()
        assert (ratio >= 1 / upper).all()

        ## TODO: after adjusting the box, it is possible that we violate prevously valid constraints wrt.
        ## the object box or wrt the containing image. The ratio limits need to be bound based on these constraints
        ## before applying randomness
    else:
        ratio = 1.0

    target_y = target_y * ratio
    target_x = target_x / ratio  # np.ones_like(ratio)
    start_x, end_x = random_seg_start(
        b.x1, b.x2, target_x, b.im_width, off_center_range=off_center_range, n=n
    )
    start_y, end_y = random_seg_start(
        b.y1, b.y2, target_y, b.im_height, off_center_range=off_center_range, n=n
    )

    assert ((bw > target_x) | (start_x <= b.x1)).all()
    assert ((bw > target_x) | (end_x >= b.x2)).all()

    return pd.DataFrame({"x1": start_x, "x2": end_x, "y1": start_y, "y2": end_y})


def randomly_extended_crop(
    im, box, scale_range, aspect_ratio_range, off_center_range, clearance, n
):
    rbs = random_container_box(
        box, scale_range, aspect_ratio_range, off_center_range, clearance, n=n
    )
    crs = []
    for cb in rbs.itertuples():
        cr = im.crop((cb.x1, cb.y1, cb.x2, cb.y2))
        crs.append(cr)
    return crs


def process_crops(crs, tx, embedding):
    if len(crs) == 0:
        return np.zeros((0, 512))

    tensors = []
    for cr in crs:
        cr = cr.resize((224, 224), resample=3)
        ts = tx(cr)
        tensors.append(ts)

    allts = torch.stack(tensors)

    emvecs = []
    bs = 20
    for i in range(0, len(allts), bs):
        batch = allts[i : i + bs]
        embs = embedding.from_image(preprocessed_image=batch, pooled="bypass")
        emvecs.append(embs)

    embs = np.concatenate(emvecs)
    return embs


import time

_clip_tx = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
        lambda x: x.type(torch.float16),
    ]
)

from .indices.multiscale.multiscale_index import box_iou


def fill_imdata(imdata: Imdata, box_data: pd.DataFrame, b: BenchParams):
    imdata = imdata.copy()
    rows = box_data[box_data.dbidx == imdata.dbidx]
    if rows.shape[0] > 0:

        positives = rows[rows.category == b.ground_truth_category]
        positives = positives.assign(marked_accepted=True)

        # find the anotations that overlap with activation boxes
        if b.provide_textual_feedback:
            negatives = rows[rows.category != b.ground_truth_category]
            activation_df = pd.DataFrame([act.box.dict() for act in imdata.activations])
            ious = box_iou(negatives, activation_df)
            highlighted_area = np.sum(ious, axis=1)
            annotated_negatives = negatives[highlighted_area > 0.1]
            annotated_negatives = annotated_negatives.assign(marked_accepted=False)
            feedback_df = pd.concat(
                [positives, annotated_negatives], axis=0, ignore_index=True
            )
        else:
            feedback_df = positives

        feedback_df = feedback_df[
            ["x1", "x2", "y1", "y2", "description", "marked_accepted"]
        ]

        ## drop some boxes based on b.box_drop_prob
        rnd = np.random.rand(feedback_df.shape[0])
        kept_rows = feedback_df[rnd >= b.box_drop_prob]
        boxes = [Box(**b) for b in kept_rows.to_dict(orient="records")]
    else:
        boxes = []
    imdata.boxes = boxes
    return imdata

import pyroaring as pr

def benchmark_loop(
    *,
    session: Session,
    subset: pr.FrozenBitMap,
    box_data: pd.DataFrame,
    b: BenchParams,
    p: SessionParams,
):
    def annotation_fun(cat):
        dataset_name = p.index_spec.d_name
        term = category2query(dataset_name, cat)
        return b.query_template.format(term)

    box_data = box_data.assign(description=box_data.category.map(annotation_fun))
    all_box_data = box_data
    box_data = box_data[box_data.category == b.ground_truth_category]
    positives = pr.FrozenBitMap(box_data.dbidx.values)

    assert positives.intersection(subset) == positives, "index mismatch"

    if b.max_results is not None:
        max_results = min(len(positives), b.max_results)
    else:
        max_results = len(positives)

    total_results = 0
    total_seen = 0
    seen_dbidxs = pr.BitMap()

    session.set_text(b.qstr)
    for batch_num in tqdm(range(1, b.n_batches + 1), leave=False, disable=True):
        print(f"iter {batch_num}")
        idxbatch = session.next()

        for idx in idxbatch:
            assert idx in subset, f'returned a dbidx outside of range'
            assert idx not in seen_dbidxs, f'returned a repeated dbidx'
            seen_dbidxs.add(idx)

        if len(idxbatch) == 0:
            break

        s = copy.deepcopy(session.get_state())
        last_batch = s.gdata[-1]
        for j, imdata in enumerate(last_batch):
            if b.provide_textual_feedback:
                last_batch[j] = fill_imdata(imdata, all_box_data, b)
            else:
                last_batch[j] = fill_imdata(imdata, box_data, b)

        session.update_state(s)
        batch_pos = np.array([is_image_accepted(imdata) for imdata in last_batch])
        total_results += batch_pos.sum()
        total_seen += idxbatch.shape[0]

        if (
            total_results >= max_results
        ):  # one batch may have more than 1, so it could go beyond desired limit
            print(
                f"Found {total_results} (>= limit of {max_results}) for {b.ground_truth_category} after {batch_num} batches. stopping..."
            )
            break

        if batch_num == b.n_batches:
            print(f"iter {batch_num} = {b.n_batches}. ending...")
            break

        if b.max_feedback is None or (batch_num + 1) * p.batch_size <= b.max_feedback:
            print("refining...")
            session.refine()

    return dict(nfound=int(total_results), nseen=int(total_seen))


def run_on_actor(br, tup):
    return br.run_loop.remote(*tup)


from .progress_bar import tqdm_map

import os
import string
import time
from .util import reset_num_cpus

from contextlib import redirect_stderr, redirect_stdout

class BenchRunner(object):
    def __init__(
        self, seesaw_root, results_dir, num_cpus: int = None, redirect_output=True
    ):
        assert os.path.isdir(results_dir)
        if num_cpus is not None:
            reset_num_cpus(num_cpus)

        vls_init_logger()
        self.gdm = GlobalDataManager(seesaw_root)
        self.results_dir = results_dir
        random.seed(int(f"{time.time_ns()}{os.getpid()}"))
        self.redirect_output = redirect_output

    def ready(self):
        return True

    def run_loop(self, b: BenchParams, p: SessionParams):
        start = time.time()
        random_suffix = "".join(
            [random.choice(string.ascii_lowercase) for _ in range(10)]
        )
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f"{self.results_dir}/session_{timestamp}_{random_suffix}"
        os.mkdir(output_dir)
        print(f"saving output to {output_dir}")
        summary = BenchSummary(
            bench_params=b, output_dir=output_dir, session_params=p, timestamp=timestamp, result=None
        )
        output_path = f"{output_dir}/summary.json"
        def closure():
            try:
                json.dump(summary.dict(), open(output_path, "w"), indent=3)

                ## also place them in log for convenience
                bench_params = json.dumps(b.dict(), indent=3)
                session_params = json.dumps(p.dict(), indent=3)
                print('bench_params')
                print(bench_params)
                print('session_params')
                print(session_params)

                ret = make_session(self.gdm, p)
                ds = ret['dataset']
                boxes, qgt = ds.load_ground_truth()
                gtseries = qgt[b.ground_truth_category]

                print("session built... now runnning loop")
                run_info = benchmark_loop(
                    session=ret["session"],
                    box_data=boxes,
                    subset=pr.BitMap(ds.file_meta.index.values),
                    b=b,
                    p=p,
                )
                print("loop done... now saving results")
                session = ret["session"]
                summary.result = BenchResult(
                    ntotal=(gtseries > 0).sum(),
                    nimages=gtseries.shape[0],
                    session=session.get_state(),
                    run_info=run_info,
                    total_time=time.time() - start,
                )

                json.dump(summary.dict(), open(output_path, "w"), indent=3)
            except Exception as exception:
                print(f'{exception=}', file=sys.stderr)
                raise exception
            
        if self.redirect_output:
            with  open(f"{output_dir}/output.log", "w") as output_log:
                with redirect_stdout(output_log), redirect_stderr(output_log):
                    closure()
        else:
            closure()

        return output_dir

import glob


def get_metric_summary(res: BenchResult):
    session = res.session
    curr_idx = 0
    hit_indices = []
    for ent in session.gdata:
        for imdata in ent:
            if is_image_accepted(imdata):
                hit_indices.append(curr_idx)
            curr_idx += 1
    index_set = pr.BitMap(hit_indices)
    assert len(index_set) == len(hit_indices)
    return dict(
        hit_indices=np.array(index_set),
        nseen=curr_idx,
        nimages=res.nimages,
        ntotal=res.ntotal,
        total_time=res.total_time,
    )


def parse_batch(batch):
    acc = []
    for (path, b) in batch:
        try:
            obj = json.loads(b)
        except json.decoder.JSONDecodeError:
            obj = {}

        obj["session_path"] = path[: -len("summary.json")]
        acc.append(obj)

    return acc


from ray.data.datasource import FastFileMetadataProvider
def load_session_data(base_dir, parallelism=-1):
    summary_paths = glob.glob(base_dir + "/**/summary.json", recursive=True)
    r = ray.data.read_binary_files(summary_paths, include_paths=True, 
                meta_provider=FastFileMetadataProvider(), parallelism=parallelism).lazy()
    res = r.map_batches(parse_batch, batch_size=20)
    return res


def process_dict(obj, mode="benchmark"):
    assert mode in ["benchmark", "session"]
    if len(obj) != 1:
        bs = BenchSummary(**obj)
        b = bs.bench_params
        s = bs.session_params
        if s.method_config == {}:
            s.method_config = None
        
        r = bs.result
        res = {**b.dict(), **s.index_spec.dict(), **s.dict()}
        res['session_params'] = s.dict()
        res['bench_params'] = b.dict()

        if bs.result is not None:
            summary = get_metric_summary(bs.result)
            res.update(summary)
    else:
        res = obj

    res["session_path"] = obj["session_path"]
    return res

from .util import as_batch_function


def _summarize(res, parallel):
    if parallel:
        acc = res.map_batches(as_batch_function(process_dict), batch_size=20).take_all()
    else:
        acc = []
        for obj in tqdm(res.iter_rows()):
            obj2 = process_dict(obj)
            acc.append(obj2)

    return pd.DataFrame.from_records(acc)


def get_all_session_summaries(base_dir, force_recompute=False, parallel=True):
    sumpath = base_dir + "/summary.parquet"
    if not os.path.exists(sumpath) or force_recompute:
        res = load_session_data(base_dir)
        df = _summarize(res, parallel=parallel)
        df.to_parquet(sumpath)

    return pd.read_parquet(sumpath)


def compute_stats(summ):
    summ = summ[~summ.ntotal.isna()]
    summ = summ.reset_index(drop=True)

    nums = summ[["batch_size", "nseen", "ntotal", "max_results"]]
    all_mets = []
    for tup in nums.itertuples():
        mets = compute_metrics(
            hit_indices=summ.hit_indices.iloc[tup.Index],
            nseen=int(tup.nseen),
            batch_size=int(tup.batch_size),
            ntotal=int(tup.ntotal),
            max_results=tup.max_results,
        )
        all_mets.append(mets)

    metrics = pd.DataFrame(all_mets)
    stats = pd.concat([summ, metrics], axis=1)
    return stats


from .basic_types import IndexSpec
from .dataset_search_terms import category2query
import numpy as np
import math

from .configs import get_session_params

def get_bench_params(b_template, name, sample_id, dataset, category):
    copy.deepcopy(b_template)

    update_b = {}
    term = category2query(dataset, category)
    qstr = b_template["query_template"].format(term)
    update_b["qstr"] = qstr
    update_b["ground_truth_category"] = category
    
    b = BenchParams(
        **{
            **b_template,
            **update_b,
            'name':name,
            'sample_id':sample_id,
        }
    )
    return b


def generate_benchmark_configs(
    gdm: GlobalDataManager,
    datasets,
    base_configs,
    s_template: SessionParams,
    b_template: BenchParams,
    max_classes_per_dataset=math.inf,
):
    ans = []
    avail_datasets = gdm.list_datasets()
    for ddict in datasets:
        if isinstance(ddict , dict):
            dataset_name = ddict['name']
            cats = ddict.get('categories', [])
        else:
            dataset_name = ddict
            cats = []

        assert dataset_name in avail_datasets
        ds = gdm.get_dataset(dataset_name)        
        classes = ds.load_eval_categories()
        if cats == []:
            cats = classes

        for i, category in enumerate(cats):
            assert category in classes
            if i == max_classes_per_dataset:
                break

            for i,config in enumerate(base_configs):
                index_meta = dict(d_name=dataset_name, c_name=(category if dataset_name == 'lvis' else None))
                s = get_session_params(s_template, config=config, index_meta=index_meta)
                b = get_bench_params(b_template, name=config['name'], sample_id=config['sample_id'], dataset=dataset_name, category=category)
                ans.append((b, s))

    return ans

def make_bench_actors(
    *, bench_constructor_args, actor_options, num_actors=None, timeout=20
):
    RemoteBenchRunner = ray.remote(BenchRunner)  # may update the definiton

    ready = {}

    if num_actors is None:
        resources = actor_options
        avail_res = ray.available_resources()
        mem_ceiling = math.ceil(avail_res["memory"] / resources["memory"])
        cpu_ceiling = math.ceil(avail_res["CPU"] / resources["num_cpus"])
        num_actors = min(mem_ceiling, cpu_ceiling)

    print(f"will try making {num_actors}")
    for i in range(num_actors):
        options = actor_options.copy()
        del options["name"]
        # options['name'] = f'{actor_options["name"]}_{i}'
        a = RemoteBenchRunner.options(**options).remote(**bench_constructor_args)
        ready[a.ready.remote()] = a

    actors = []
    print(f"waiting for actors to be ready or for timeout {timeout} s")
    done, not_done = ray.wait(
        list(ready.keys()), num_returns=len(ready), timeout=timeout
    )
    for d in done:
        actors.append(ready[d])

    ## clean up other actors
    for d in not_done:
        ray.kill(ready[d])

    return actors


def parallel_run(*, actors, tups):
    res = []
    tqdm_map(actors, run_on_actor, tups, res=res)
