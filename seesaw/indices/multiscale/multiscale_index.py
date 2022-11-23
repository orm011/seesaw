from ray.data.extensions import TensorArray

import torchvision
from torchvision import transforms as T

import numpy as np
import pandas as pd
from seesaw.dataset_tools import *
from torch.utils.data import DataLoader
import math
from tqdm.auto import tqdm
import torch
from seesaw.labeldb import LabelDB
from seesaw.query_interface import  AccessMethod, InteractiveQuery

from seesaw.models.embeddings import make_clip_transform, ImTransform, XEmbedding
import pyroaring as pr
from operator import itemgetter
import PIL
from seesaw.vector_index import VectorIndex
import math
import annoy
from seesaw.definitions import resolve_path
import os


def _postprocess_results(acc):
    flat_acc = {
        "iis": [],
        "jjs": [],
        "dbidx": [],
        "vecs": [],
        "zoom_factor": [],
        "zoom_level": [],
    }
    flat_vecs = []

    # {'accs':accs, 'sf':sf, 'dbidx':dbidx, 'zoom_level':zoom_level}
    for item in acc:
        acc0, sf, dbidx, zl = itemgetter("accs", "sf", "dbidx", "zoom_level")(item)
        acc0 = acc0.squeeze(0)
        acc0 = acc0.transpose((1, 2, 0))

        iis, jjs = np.meshgrid(
            range(acc0.shape[0]), range(acc0.shape[1]), indexing="ij"
        )
        # iis = iis.reshape(-1, acc0)
        iis = iis.reshape(-1)
        jjs = jjs.reshape(-1)
        acc0 = acc0.reshape(-1, acc0.shape[-1])
        imids = np.ones_like(iis) * dbidx
        zf = np.ones_like(iis) * (1.0 / sf)
        zl = np.ones_like(iis) * zl

        flat_acc["iis"].append(iis)
        flat_acc["jjs"].append(jjs)
        flat_acc["dbidx"].append(imids)
        flat_acc["vecs"].append(acc0)
        flat_acc["zoom_factor"].append(zf)
        flat_acc["zoom_level"].append(zl)

    flat = {}
    for k, v in flat_acc.items():
        flat[k] = np.concatenate(v)

    vecs = flat["vecs"]
    del flat["vecs"]

    vec_meta = pd.DataFrame(flat)
    vecs = vecs.astype("float32")
    vecs = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-6)
    vec_meta = vec_meta.assign(file_path=item["file_path"])

    vec_meta = vec_meta.assign(vectors=TensorArray(vecs))
    return vec_meta


def preprocess_ds(localxclip, ds, debug=False):
    txds = TxDataset(ds, tx=pyramid_tx(non_resized_transform(224)))
    acc = []
    if debug:
        num_workers = 0
    else:
        num_workers = 4
    for dbidx, tup in enumerate(
        tqdm(
            DataLoader(
                txds,
                num_workers=num_workers,
                shuffle=False,
                batch_size=1,
                collate_fn=lambda x: x,
            ),
            total=len(txds),
        )
    ):
        [(ims, sfs)] = tup
        for zoom_level, (im, sf) in enumerate(zip(ims, sfs), start=1):
            accs = localxclip.from_image(preprocessed_image=im, pooled=False)
            acc.append((accs, sf, dbidx, zoom_level))

    return _postprocess_results(acc)


def pyramid_centered(im, i, j):
    cy = (i + 1) * 112.0
    cx = (j + 1) * 112.0
    scales = [112, 224, 448]
    crs = []
    w, h = im.size
    for s in scales:
        tup = (
            np.clip(cx - s, 0, w),
            np.clip(cy - s, 0, h),
            np.clip(cx + s, 0, w),
            np.clip(cy + s, 0, h),
        )
        crs.append(im.crop(tup))
    return crs


def zoom_out(im: PIL.Image, factor=0.5, abs_min=224):
    """
    returns image one zoom level out, and the scale factor used
    """
    w, h = im.size
    mindim = min(w, h)
    target_size = max(math.floor(mindim * factor), abs_min)
    if (
        target_size * math.sqrt(factor) <= abs_min
    ):  # if the target size is almost as large as the image,
        # jump to that scale instead
        target_size = abs_min

    target_factor = target_size / mindim
    target_w = max(
        math.floor(w * target_factor), 224
    )  # corrects any rounding effects that make the size 223
    target_h = max(math.floor(h * target_factor), 224)

    im1 = im.resize((target_w, target_h))
    assert min(im1.size) >= abs_min
    return im1, target_factor


def rescale(im, scale, min_size):
    (w, h) = im.size
    target_w = max(math.floor(w * scale), min_size)
    target_h = max(math.floor(h * scale), min_size)
    return im.resize(size=(target_w, target_h), resample=PIL.Image.BILINEAR)


def pyramid(im, factor=0.71, abs_min=224):
    ## if im size is less tha the minimum, expand image to fit minimum
    ## try following: orig size and abs min size give you bounds
    assert factor < 1.0
    factor = 1.0 / factor
    size = min(im.size)
    end_size = abs_min
    start_size = max(size, abs_min)

    start_scale = start_size / size
    end_scale = end_size / size

    ## adjust start scale
    ntimes = math.ceil(math.log(start_scale / end_scale) / math.log(factor))
    start_size = math.ceil(math.exp(ntimes * math.log(factor) + math.log(abs_min)))
    start_scale = start_size / size
    factors = np.geomspace(
        start=start_scale, stop=end_scale, num=ntimes + 1, endpoint=True
    ).tolist()
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
        w1, h1 = im1.size
        spare_h = h1 % target_divisor
        spare_w = w1 % target_divisor
        im1 = im1.crop((0, 0, w1 - spare_w, h1 - spare_h))
        return im1

    return fun


class TrimEdge:
    def __init__(self, target_divisor=112):
        self.target_divisor = target_divisor

    def __call__(self, im1):
        w1, h1 = im1.size
        spare_h = h1 % self.target_divisor
        spare_w = w1 % self.target_divisor
        im1 = im1.crop((0, 0, w1 - spare_w, h1 - spare_h))
        return im1


def torgb(image):
    return image.convert("RGB")


def tofloat16(x):
    return x.type(torch.float16)


def non_resized_transform(base_size):
    return ImTransform(
        visual_xforms=[torgb],
        tensor_xforms=[
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
            # tofloat16
        ],
    )


class PyramidTx:
    def __init__(self, tx, factor, min_size):
        self.tx = tx
        self.factor = factor
        self.min_size = min_size

    def __call__(self, im):
        ims, sfs = pyramid(im, factor=self.factor, abs_min=self.min_size)
        ppims = []
        for im in ims:
            ppims.append(self.tx(im))

        return ppims, sfs


def pyramid_tx(tx):
    def fn(im):
        ims, sfs = pyramid(im)
        ppims = []
        for im in ims:
            ppims.append(tx(im))

        return ppims, sfs

    return fn


def augment_score(db, tup, qvec):
    im = db.raw[tup.dbidx]
    ims = pyramid(im, tup.iis, tup.jjs)
    tx = make_clip_transform(n_px=224, square_crop=True)
    vecs = []
    for im in ims:
        pim = tx(im)
        emb = db.embedding.from_image(preprocessed_image=pim.float())
        emb = emb / np.linalg.norm(emb, axis=-1)
        vecs.append(emb)

    vecs = np.concatenate(vecs)
    # print(np.linalg.norm(vecs,axis=-1))
    augscore = (vecs @ qvec.reshape(-1)).mean()
    return augscore


import torchvision.ops

from ...box_utils import box_iou

def augment_score2(tup, vec_meta, vecs, *, agg_method, rescore_method, aug_larger):
    assert tup.shape[0] == 1
    assert callable(rescore_method)

    if agg_method == "plain_score":
        return tup.score.values[0]

    vec_meta = vec_meta.reset_index(drop=True)
    ious, containments = box_iou(tup, vec_meta, return_containment=True)

    vec_meta = vec_meta.assign(iou=ious.reshape(-1), containments=containments.reshape(-1))
    max_boxes = vec_meta.groupby("zoom_level").iou.idxmax()
    # largest zoom level means zoomed out max
    relevant_meta = vec_meta.iloc[max_boxes.values]
    relevant_mask = (
        relevant_meta.iou > 0
    )  # there should be at least some overlap for it to be relevant

    zl = int(tup.zoom_level.values[0])
    if aug_larger == 'all':
        pass ## already have rel mask
    elif aug_larger == 'greater':
        relevant_mask = relevant_mask & (relevant_meta.zoom_level >= zl)
    elif aug_larger == 'adjacent':
        relevant_mask = relevant_mask & (relevant_meta.zoom_level.isin([zl, zl+1]))
    else:
        assert False, f"unknown aug_larger {aug_larger}"

    max_boxes = max_boxes[relevant_mask.values]
    rel_vecs = vecs[max_boxes]

    if agg_method == "avg_score":
        sc = rescore_method(rel_vecs)
        ws = np.ones_like(sc)
        fsc = ws.reshape(-1) @ sc.reshape(-1)
        fsc = fsc / ws.sum()
        return fsc
    elif agg_method == "avg_vector":
        merged_vec = rel_vecs.mean(axis=0, keepdims=True)
        merged_vec = merged_vec / np.linalg.norm(merged_vec)
        return rescore_method(merged_vec)
    else:
        assert False, f"unknown agg_method {agg_method}"



def get_boxes(vec_meta):
    if "x1" in vec_meta.columns:
        return vec_meta[["x1", "x2", "y1", "y2"]]

    y1 = vec_meta.iis * 112
    y2 = y1 + 224
    x1 = vec_meta.jjs * 112
    x2 = x1 + 224
    factor = vec_meta.zoom_factor
    boxes = vec_meta.assign(
        **{"x1": x1 * factor, "x2": x2 * factor, "y1": y1 * factor, "y2": y2 * factor}
    )[["x1", "y1", "x2", "y2"]]
    boxes = boxes.astype(
        "float32"
    )  ## multiplication makes this type double but this is too much.
    return boxes

from seesaw.box_utils import left_iou_join

def get_pos_negs_all_v3(label_db: LabelDB, vec_meta: pd.DataFrame):
    idxs = label_db.get_seen()
    vec_meta = vec_meta[vec_meta.dbidx.isin(idxs)]
    boxdf = label_db.get_box_df()
    vec_meta_new = left_iou_join(vec_meta, boxdf)
    vec_meta_new = vec_meta_new.assign(ys = (vec_meta_new.max_iou > 0).astype('float'))
    pos = vec_meta_new.index[vec_meta_new.ys > 0].values
    neg = vec_meta_new.index[vec_meta_new.ys == 0].values
    return pos, neg

def get_pos_negs_all_v2(label_db: LabelDB, vec_meta: pd.DataFrame):
    idxs = label_db.get_seen()
    pos = [np.array([], dtype=vec_meta.index.values.dtype)]
    neg = [np.array([], dtype=vec_meta.index.values.dtype)]
    
    vec_meta = vec_meta[vec_meta.dbidx.isin(idxs)]

    for idx, acc_vecs in vec_meta.groupby('dbidx'):
        label_boxes = label_db.get(idx, format="df")
        if label_boxes.shape[0] == 0:
            ## every vector is a negative example in this case
            neg.append(acc_vecs.index.values)
            pos.append(np.array([], dtype=acc_vecs.index.values.dtype))

        ious = box_iou(label_boxes, acc_vecs)
        total_iou = ious.sum(axis=0)
        negatives = total_iou == 0
        negvec_positions = acc_vecs.index[negatives].values

        # get the highest iou positives for each
        max_ious_id = np.argmax(ious, axis=1)
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
    t = annoy.AnnoyIndex(512, "dot")
    for i in range(len(vecs)):
        t.add_item(i, vecs[i])
    t.build(n_trees=100)  # tested 100 on bdd, works well, could do more.
    t.save(file_name)
    u = annoy.AnnoyIndex(512, "dot")
    u.load(file_name)  # verify can load.
    return u


def filter_mask(meta, min_level_inclusive):
    gpmax = meta.groupby("dbidx").zoom_level.max().rename("zoom_level_max")
    aug_meta = pd.merge(meta, gpmax, left_on="dbidx", right_index=True)
    is_max = aug_meta.zoom_level == aug_meta.zoom_level_max
    is_larger = aug_meta.zoom_level >= min_level_inclusive
    mask = is_max | is_larger
    return mask.values

import json

def score_frame(*, frame_meta, agg_method, rescore_method, aug_larger):
    topscore = frame_meta.score.max()
    tup = frame_meta[frame_meta.score == topscore].head(n=1) # sometimes there are more than one
    score = augment_score2(tup, frame_meta, vecs=frame_meta.vectors.to_numpy(), 
                               agg_method=agg_method, rescore_method=rescore_method, aug_larger=aug_larger)

    return tup.assign(score=score)

from seesaw.box_utils import box_join
from scipy.special import softmax

def score_frame2(meta_df, **aug_options):
    aug_larger=aug_options['aug_larger']
    aug_weight=aug_options.get('aug_weight', 'level_max')
    agg_method=aug_options['agg_method']
    
    if agg_method == 'plain_score':
        return meta_df.query('score == score.max()').head(n=1)
    

    meta_df = meta_df.reset_index(drop=True)
    mdf = meta_df[['x1', 'x2', 'y1', 'y2', 'zoom_level', 'score']]
    joined = box_join(mdf, mdf)
    
    if aug_larger == 'greater':
        joined = joined.query('zoom_level_right >= zoom_level_left')
    elif aug_larger == 'adjacent':
        joined = joined.query('zoom_level_right == zoom_level_left')
    elif aug_larger == 'all':
        pass
    else:
        assert False
        
    def weighted_score(gp):
        weights = softmax(gp.cont.values)
        score = weights @ gp.score_right.values
        return score
    
    joined = joined.reset_index(drop=True)
    
    if aug_weight == 'level_max':
        idxmaxes = joined.groupby(['iloc_left', 'zoom_level_right']).iou.idxmax()
        max_only = joined.iloc[idxmaxes.values]
        all_scores = max_only.groupby('iloc_left').score_right.mean()
    elif aug_weight == 'cont_weighted':
        all_scores = joined.groupby('iloc_left').apply(weighted_score)
    else:
        assert False
        
    meta_df = meta_df.assign(unadjusted_score=meta_df.score, score=all_scores)
    return meta_df.query('score == score.max()').head(n=1)

def _get_top_approx(vector, *, vector_meta, vec_index, exclude, topk):
    i = 0
    deltak = topk * 10
    while True:
        if i > 1:
            print("warning, we are looping too much. adjust initial params?")

        vec_idxs, vec_scores = vec_index.query(vector, top_k=deltak)
        found_idxs = pr.BitMap(vector_meta['dbidx'].values[vec_idxs])
        newidxs = found_idxs.difference(exclude)
        if len(newidxs) >= topk:
            break
        else:
            deltak = deltak * 2
            i += 1

    return vec_idxs, vec_scores

def _get_top_exact(vector, *, vectors):
    scores = vectors @ vector.reshape(-1)
    vec_idxs = np.argsort(-scores)
    vec_scores = scores[vec_idxs]

    return vec_idxs, vec_scores

def distinct_topk_positions(dbidxs, topk): 
    """returns the position of the topk distinct dbidxs within the array."""
    _, index = np.unique(dbidxs, return_index=True)
    return np.sort(index)[:topk]

def test_distinct_topk_positions():
    ex_dbidx = np.array([10,11,11,12,12,12,13,13])
    expect = np.array([0,1,3,6])
    k = 2
    ans = distinct_topk_positions(ex_dbidx, k)
    assert (ans == expect[:k]).all()

def _get_top_dbidxs(*, vec_idxs, scores, vector_meta, exclude, topk):
    """ return the topk non-excluded dbidxs 
    """
    dbidx = vector_meta.dbidx.iloc[vec_idxs]
    mask = (~dbidx.isin(exclude)).values
    new_dbidx = dbidx[mask].values
    new_scores = scores[mask]

    pos = distinct_topk_positions(new_dbidx, topk=topk)
    df = pd.DataFrame({'dbidx':new_dbidx[pos], 'max_score':new_scores[pos]})    
    return df



class MultiscaleIndex(AccessMethod):
    """implements a two stage lookup"""

    def __init__(
        self,
        *,
        embedding: XEmbedding,
        vectors: np.ndarray,
        vector_meta: pd.DataFrame,
        vec_index=None,
        min_zoom_level=1,
        path : str = None,
    ):
        self.embedding = embedding
        self.path = path

        if min_zoom_level == 1:
            self.vectors = vectors
            self.vector_meta = vector_meta
            self.vec_index = vec_index
            self.all_indices = pr.FrozenBitMap(self.vector_meta.dbidx.values)
        else:  # filter out lowest zoom level
            print("WARNING: filtering out min_zoom_level")
            mask = filter_mask(vector_meta, min_level_inclusive=min_zoom_level)
            self.vector_meta = vector_meta[mask].reset_index(drop=True)
            self.vectors = vectors[mask]

            self.vec_index = None  # no index constructed here
            self.all_indices = pr.FrozenBitMap(self.vector_meta.dbidx.values)

    @staticmethod
    def from_path(index_path: str, *, use_vec_index=True, **options):
        from ...services import get_parquet, get_model_actor

        index_path = resolve_path(index_path)
        options = json.load(open(f'{index_path}/info.json'))
        model_path = options['model'] #os.readlink(f"{index_path}/model")
        embedding = get_model_actor(model_path)
        cached_meta_path = f"{index_path}/vectors.sorted.cached"

        if use_vec_index:
            fullpath = f"{index_path}/vectors.annoy"
            print(f"looking for vector index in {fullpath}")
            assert os.path.exists(fullpath)
            vec_index = VectorIndex(load_path=fullpath, prefault=True)
        else:
            print('NOTE: not using vector index')
            vec_index = None

        assert os.path.exists(cached_meta_path)
        df: pd.DataFrame = get_parquet(cached_meta_path).reset_index(drop=True)
        # assert df["order_col"].is_monotonic_increasing, "sanity check"

        fine_grained_meta = df[
            ["dbidx", "zoom_level", "x1", "y1", "x2", "y2"]
        ]
        fine_grained_embedding = df["vectors"].values.to_numpy()

        return MultiscaleIndex(
            embedding=embedding,
            vectors=fine_grained_embedding,
            vector_meta=fine_grained_meta,
            vec_index=vec_index,
            path = index_path
        )

    def get_knng(self, path=None):
        from ...research.knn_methods import KNNGraph
        if path is None:
            path = ''
            
        knng = KNNGraph.from_file(f'{self.path}/knn_graph/{path}')
        return knng

    def string2vec(self, string: str):
        init_vec = self.embedding.from_string(string=string)
        init_vec = init_vec / np.linalg.norm(init_vec)
        return init_vec

    def score(self, vec):
        return self.vectors @ vec.reshape(-1)

    def __len__(self):
        return len(self.all_indices)


    def _query_prelim(self, *, vector, topk_dbidx, exclude_dbidx=None, force_exact=False):
        if exclude_dbidx is None:
            exclude_dbidx = pr.BitMap([])

        included_dbidx = pr.BitMap(self.all_indices).difference(exclude_dbidx)
        
        if len(included_dbidx) <= topk_dbidx:
            topk_dbidx = len(included_dbidx)

        if topk_dbidx == 0:
            print("no dbidx included")
            return [], [], []

        if self.vec_index is None or force_exact:
            vec_idxs, vec_scores = _get_top_exact(vector, vectors=self.vectors)
        else:
            vec_idxs, vec_scores = _get_top_approx(vector, vector_meta=self.vector_meta, 
                                    vec_index=self.vec_index, exclude=exclude_dbidx, topk=topk_dbidx)

        dbidxs = _get_top_dbidxs(vec_idxs=vec_idxs, scores=vec_scores, vector_meta=self.vector_meta, 
                                exclude=exclude_dbidx, topk=topk_dbidx)
        return dbidxs

    def query(
        self,
        *,
        vector,
        topk,
        shortlist_size,
        exclude=None,
        force_exact=False,
        **kwargs,
    ):
        if shortlist_size is None:
            shortlist_size = topk * 5

        if shortlist_size < topk * 5:
            print(
                f"Warning: shortlist_size parameter {shortlist_size} is small compared to topk param {topk}, you may consider increasing it"
            )

        qvec = vector
        candidate_df = self._query_prelim(
            vector=qvec,
            topk_dbidx=shortlist_size,
            exclude_dbidx=exclude,
            force_exact = force_exact
        )


        candidate_id = pr.BitMap(candidate_df['dbidx'].values)
        ilocs = np.where(self.vector_meta.dbidx.isin(candidate_id))[0]
        fullmeta : pd.DataFrame = self.vector_meta.iloc[ilocs]
        vectors = self.vectors[ilocs]
        scores = vectors @ qvec.reshape(-1)
        fullmeta = fullmeta.assign(score=scores, vectors=TensorArray(vectors))
        return rescore_candidates(fullmeta, topk, **kwargs)


    def new_query(self):
        return BoxFeedbackQuery(self)

    def get_data(self, dbidx) -> pd.DataFrame:
        vmeta = self.vector_meta[self.vector_meta.dbidx == dbidx]
        vectors = self.vectors[vmeta.index]

        return vmeta.assign(vectors=TensorArray(vectors))

    def subset(self, indices: pr.BitMap) -> AccessMethod:
        mask = self.vector_meta.dbidx.isin(indices)
        if mask.all():
            return self
        else:
            if self.vec_index is not None:
                print(
                    "warning: after subsetting we lose ability to use pre-built index"
                )

        vector_meta = self.vector_meta[mask].reset_index(drop=True)
        vectors = self.vectors[mask]
        return MultiscaleIndex(
            embedding=self.embedding,
            vectors=vectors,
            vector_meta=vector_meta,
            vec_index=None,
        )


def rescore_candidates(fullmeta, topk, **kwargs):
        fullmeta = fullmeta.reset_index(drop=True) # for some files (coarse) dbidx is also the index name
        ## which causes groupby to fail.
        nframes = fullmeta.dbidx.unique().shape[0]
        dbidxs = np.zeros(nframes) * -1
        dbscores = np.zeros(nframes)
        activations = []

        ## for each frame, compute augmented scores for each tile and record max
        for i, (dbidx, frame_meta) in enumerate(fullmeta.groupby("dbidx")):
            dbidxs[i] = dbidx
            tup = score_frame2(frame_meta, **kwargs)

            frame_activations = tup[
                ["x1", "y1", "x2", "y2", "dbidx", "score"]
            ]

            dbscores[i] = tup.score.iloc[0]
            activations.append(frame_activations)

        topkidx = np.argsort(-dbscores)[:topk]
        return {
            "dbidxs": dbidxs[topkidx].astype("int"),
            "activations": [activations[idx] for idx in topkidx]
        }



def add_iou_score(box_df: pd.DataFrame, roi_box_df: pd.DataFrame):
    """assumes vector_data is a df with box information"""
    ious = box_iou(box_df, roi_box_df)

    best_match = np.argmax(ious, axis=1)  # , .idxmax(axis=1)
    best_iou = np.max(ious, axis=1)
    box_df = box_df.assign(best_box_iou=best_iou, best_box_idx=best_match)
    return box_df


class BoxFeedbackQuery(InteractiveQuery):
    def __init__(self, db):
        super().__init__(db)
        # self.acc_pos = []
        # self.acc_neg = []

    def getXy(self, get_positions=False):
        pos, neg = get_pos_negs_all_v3(self.label_db, self.index.vector_meta)
        if get_positions:
            return pos, neg

        ## we are currently ignoring these positives
        # self.acc_pos.append(batchpos)
        # self.acc_neg.append(batchneg)
        # pos = pr.BitMap.union(*self.acc_pos)
        # neg = pr.BitMap.union(*self.acc_neg)

        allpos = self.index.vectors[pos]
        Xt = np.concatenate([allpos, self.index.vectors[neg]])
        yt = np.concatenate([np.ones(len(allpos)), np.zeros(len(neg))])
        return Xt, yt
        # not really valid. some boxes are area 0. they should be ignored.but they affect qgt
        # if np.concatenate(acc_results).sum() > 0:
        #    assert len(pos) > 0
