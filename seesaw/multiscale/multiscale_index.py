from ray.data.extensions import TensorArray

import torchvision
from torchvision import transforms as T

import numpy as np
import pandas as pd
from ..dataset_tools import *
from torch.utils.data import DataLoader
import math
from tqdm.auto import tqdm
import torch
from ..query_interface import *

from ..models.embeddings import make_clip_transform, ImTransform, XEmbedding
from ..dataset_search_terms import *
import pyroaring as pr
from operator import itemgetter
import PIL
from ..vector_index import VectorIndex
import math
import annoy
from ..definitions import resolve_path
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

# torchvision.ops.box_iou()


def box_iou(tup, boxes):
    b1 = torch.from_numpy(
        np.stack([tup.x1.values, tup.y1.values, tup.x2.values, tup.y2.values], axis=1)
    )
    bxdata = np.stack(
        [boxes.x1.values, boxes.y1.values, boxes.x2.values, boxes.y2.values], axis=1
    )
    b2 = torch.from_numpy(bxdata)
    ious = torchvision.ops.box_iou(b1, b2)
    return ious.numpy()


def augment_score2(tup, vec_meta, vecs, *, agg_method, rescore_method):
    assert callable(rescore_method)
    vec_meta = vec_meta.reset_index(drop=True)
    ious = box_iou(tup, vec_meta)
    vec_meta = vec_meta.assign(iou=ious.reshape(-1))
    max_boxes = vec_meta.groupby("zoom_level").iou.idxmax()
    max_boxes = max_boxes.sort_index(
        ascending=True
    )  # largest zoom level (zoomed out) goes last
    relevant_meta = vec_meta.iloc[max_boxes]
    relevant_iou = (
        relevant_meta.iou > 0
    )  # there should be at least some overlap for it to be relevant
    max_boxes = max_boxes[relevant_iou.values]
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


def get_pos_negs_all_v2(dbidxs, label_db: LabelDB, vec_meta: pd.DataFrame):
    idxs = pr.BitMap(dbidxs)
    relvecs = vec_meta[vec_meta.dbidx.isin(idxs)]

    pos = []
    neg = []
    for idx in dbidxs:
        acc_vecs = relvecs[relvecs.dbidx == idx]
        acc_boxes = get_boxes(acc_vecs)
        label_boxes = label_db.get(idx, format="df")
        ious = box_iou(label_boxes, acc_boxes)
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
    ):
        self.embedding = embedding

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
    def from_path(gdm, index_subpath: str, model_name: str):
        embedding = gdm.get_model_actor(model_name)
        cached_meta_path = f"{gdm.root}/{index_subpath}/vectors.sorted.cached"

        relpath = f"{index_subpath}/vectors.annoy"
        fullpath = f"{gdm.root}/{relpath}"

        print(f"looking for vector index in {fullpath}")
        if os.path.exists(fullpath):
            print("using optimized index...")
            vec_index = VectorIndex(load_path=fullpath, prefault=True)
        else:
            print("index file not found... using vectors")
            vec_index = None

        assert os.path.exists(cached_meta_path)
        df = gdm.global_cache.read_parquet(cached_meta_path)
        assert df.order_col.is_monotonic_increasing, "sanity check"
        fine_grained_meta = df[
            ["dbidx", "order_col", "zoom_level", "x1", "y1", "x2", "y2"]
        ]
        fine_grained_embedding = df["vectors"].values.to_numpy()

        return MultiscaleIndex(
            embedding=embedding,
            vectors=fine_grained_embedding,
            vector_meta=fine_grained_meta,
            vec_index=vec_index,
        )

    @staticmethod
    def from_dir(index_path: str):
        from ..services import get_parquet, get_model_actor

        index_path = resolve_path(index_path)
        model_path = os.readlink(f"{index_path}/model")
        embedding = get_model_actor(model_path)
        cached_meta_path = f"{index_path}/vectors.sorted.cached"
        fullpath = f"{index_path}/vectors.annoy"

        print(f"looking for vector index in {fullpath}")
        if os.path.exists(fullpath):
            print("using optimized index...")
            vec_index = VectorIndex(load_path=fullpath, prefault=True)
        else:
            print("index file not found... using vectors")
            vec_index = None

        assert os.path.exists(cached_meta_path)
        df: pd.DataFrame = get_parquet(cached_meta_path)
        assert df["order_col"].is_monotonic_increasing, "sanity check"

        fine_grained_meta = df[
            ["dbidx", "order_col", "zoom_level", "x1", "y1", "x2", "y2"]
        ]
        fine_grained_embedding = df["vectors"].values.to_numpy()

        return MultiscaleIndex(
            embedding=embedding,
            vectors=fine_grained_embedding,
            vector_meta=fine_grained_meta,
            vec_index=vec_index,
        )

    def string2vec(self, string: str):
        init_vec = self.embedding.from_string(string=string)
        init_vec = init_vec / np.linalg.norm(init_vec)
        return init_vec

    def __len__(self):
        return len(self.all_indices)

    def _query_prelim(self, *, vector, topk, zoom_level, exclude=None, startk=None):
        if exclude is None:
            exclude = pr.BitMap([])

        included_dbidx = pr.BitMap(self.all_indices).difference(exclude)
        vec_meta = self.vector_meta

        if len(included_dbidx) == 0:
            print("no dbidx included")
            return [], [], []

        if len(included_dbidx) <= topk:
            topk = len(included_dbidx)

        ## want to return proposals only for images we have not seen yet...
        ## but library does not allow this...
        ## guess how much we need... and check
        def get_nns(startk, topk):
            i = 0
            deltak = topk * 100
            while True:
                if i > 1:
                    print("warning, we are looping too much. adjust initial params?")

                vec_idxs, scores = self.vec_index.query(vector, top_k=startk + deltak)
                found_idxs = pr.BitMap(vec_meta.dbidx.values[vec_idxs])

                newidxs = found_idxs.difference(exclude)
                if len(newidxs) >= topk:
                    break

                deltak = deltak * 2
                i += 1

            return vec_idxs, scores

        def get_nns_by_vector_exact():
            scores = self.vectors @ vector.reshape(-1)
            vec_idxs = np.argsort(-scores)
            return vec_idxs, scores[vec_idxs]

        if self.vec_index is not None:
            idxs, scores = get_nns(startk, topk)
        else:
            idxs, scores = get_nns_by_vector_exact()

        # work only with the two columns here bc dataframe can be large
        topscores = vec_meta[["dbidx"]].iloc[idxs]
        topscores = topscores.assign(score=scores)
        allscores = topscores

        newtopscores = topscores[~topscores.dbidx.isin(exclude)]
        scoresbydbidx = (
            newtopscores.groupby("dbidx").score.max().sort_values(ascending=False)
        )
        score_cutoff = scoresbydbidx.iloc[topk - 1]  # kth largest score
        newtopscores = newtopscores[newtopscores.score >= score_cutoff]

        # newtopscores = newtopscores.sort_values(ascending=False)
        nextstartk = (allscores.score >= score_cutoff).sum()
        nextstartk = math.ceil(
            startk * 0.8 + nextstartk * 0.2
        )  # average to estimate next
        candidates = pr.BitMap(newtopscores.dbidx)
        assert len(candidates) >= topk
        assert candidates.intersection_cardinality(exclude) == 0
        return newtopscores.index.values, candidates, allscores, nextstartk

    def query(
        self,
        *,
        vector,
        topk,
        shortlist_size,
        agg_method,
        rescore_method,
        exclude=None,
        startk=None,
        **kwargs,
    ):
        if shortlist_size is None:
            shortlist_size = topk * 5

        if shortlist_size < topk * 5:
            print(
                f"Warning: shortlist_size parameter {shortlist_size} is small compared to topk param {topk}, you may consider increasing it"
            )

        if startk is None:
            startk = len(exclude) * 10

        db = self
        qvec = vector
        meta_idx, candidate_id, allscores, nextstartk = self._query_prelim(
            vector=qvec,
            topk=shortlist_size,
            zoom_level=None,
            exclude=exclude,
            startk=startk,
        )

        fullmeta = self.vector_meta[self.vector_meta.dbidx.isin(candidate_id)]
        fullmeta = fullmeta.assign(**get_boxes(fullmeta))

        scmeta = self.vector_meta.iloc[meta_idx]
        scmeta = scmeta.assign(**get_boxes(scmeta))
        nframes = len(candidate_id)
        dbidxs = np.zeros(nframes) * -1
        dbscores = np.zeros(nframes)
        activations = []

        ## for each frame, compute augmented scores for each tile and record max
        for i, (dbidx, frame_vec_meta) in enumerate(scmeta.groupby("dbidx")):
            dbidxs[i] = dbidx
            relmeta = fullmeta[
                fullmeta.dbidx == dbidx
            ]  # get metadata for all boxes in frame.
            relvecs = db.vectors[relmeta.index.values]
            boxscs = np.zeros(frame_vec_meta.shape[0])
            for j in range(frame_vec_meta.shape[0]):
                tup = frame_vec_meta.iloc[j : j + 1]
                boxscs[j] = augment_score2(
                    tup,
                    relmeta,
                    relvecs,
                    agg_method=agg_method,
                    rescore_method=rescore_method,
                )

            frame_activations = frame_vec_meta.assign(score=boxscs)[
                ["x1", "y1", "x2", "y2", "dbidx", "score"]
            ]
            activations.append(frame_activations)
            dbscores[i] = np.max(boxscs)

        topkidx = np.argsort(-dbscores)[:topk]
        return {
            "dbidxs": dbidxs[topkidx].astype("int"),
            "nextstartk": nextstartk,
            "activations": [activations[idx] for idx in topkidx],
        }

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

    def getXy(self):
        pos, neg = get_pos_negs_all_v2(
            self.label_db.get_seen(), self.label_db, self.index.vector_meta
        )

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
