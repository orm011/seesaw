import pandas as pd
import PIL
import PIL.Image
import numpy as np
import functools
import json, os, copy
import torch, torchvision
import pyroaring as pr

# from .dataset_tools import DataFrameDataset
from .dataset_tools import TxDataset
import torchvision.models
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression


def random_resize(im, abs_min=60, abs_max=1000, rel_min=1.0 / 3, rel_max=3.0):
    (w, h) = im.size
    abs_min = min([w, h, abs_min])
    abs_max = max([w, h, abs_max])

    min_scale = max([abs_min / w, abs_min / h, rel_min])
    max_scale = min([abs_max / w, abs_max / h, rel_max])

    lower = np.log(min_scale)
    upper = np.log(max_scale)
    factor = np.exp(np.random.uniform(lower, upper))

    assert rel_min <= factor <= rel_max
    assert abs_min <= factor * w <= abs_max
    assert abs_min <= factor * h <= abs_max

    (nw, nh) = int(factor * w), int(factor * h)
    new_im = im.resize((nw, nh))
    return new_im


# def extract_rois()
def clip_boxes(img_size, boxes):
    (w, h) = img_size
    xmin = np.clip(boxes.x1, 0, w)
    xmax = np.clip(boxes.x2, 0, w)
    ymin = np.clip(boxes.y1, 0, h)
    ymax = np.clip(boxes.y2, 0, h)
    return boxes.assign(x1=xmin, y1=ymin, x2=xmax, y2=ymax)


def pad_boxes(box_data, pad_factor):
    bd = box_data[["x1", "y1", "x2", "y2"]]
    bd = bd.assign(
        midx=(bd.x1 + bd.x2) / 2,
        midy=(bd.y1 + bd.y2) / 2,
        width=(bd.x2 - bd.x1),
        height=(bd.y2 - bd.y1),
    )
    bd = bd.assign(width=bd.width * pad_factor, height=bd.height * pad_factor)
    bd2 = bd.assign(
        x1=(bd.midx - bd.width / 2),
        x2=(bd.midx + bd.width / 2),
        y1=(bd.midy - bd.height / 2),
        y2=(bd.midy + bd.height / 2),
    )
    return bd2


def extract_rois(db, ldata, pad_factor):
    rois = []
    for r in ldata:
        im = db.raw[r["dbidx"]]
        if len(r["boxes"]) > 0:
            bx = pd.DataFrame(r["boxes"])
            bx = bx.rename(
                mapper={"xmin": "x1", "xmax": "x2", "ymin": "y1", "ymax": "y2"}, axis=1
            )
            bx2 = pad_boxes(bx, pad_factor)
            bx3 = clip_boxes(im.size, bx2)
            for b in bx3.itertuples():
                roi = im.crop((b.x1, b.y1, b.x2, b.y2))
                rois.append(roi)
    return rois


def embed_rois(db, rois):
    roi_vecs = []
    for roi in rois:
        roi_vec = db.embedding.from_raw(roi)
        roi_vecs.append(roi_vec)

    roivecs = np.concatenate(
        [np.zeros((0, db.embedded.shape[1]))] + roi_vecs
    )  # in case empty
    return roivecs


def augment(rois, n=5):
    augmented_rois = []
    for roi in rois:
        (a, b) = roi.size
        # (int(b/1.5),int(a/1.5)), scale=[.8, 1.]
        t = T.Compose(
            [  # T.RandomCrop(int(a*.9), int(b*.9)),
                random_resize,
                T.RandomHorizontalFlip(),
            ]
        )
        xroi = [t(roi) for _ in range(n)]
        augmented_rois.extend(xroi)
    return augmented_rois


import pyroaring as pr


def get_panel_data(q, label_db, next_idxs):
    reslabs = []
    for (i, dbidx) in enumerate(next_idxs):
        boxes = copy.deepcopy(label_db.get(dbidx, None))
        reslabs.append(
            {
                "value": -1 if boxes is None else 1 if len(boxes) > 0 else 0,
                "id": i,
                "dbidx": int(dbidx),
                "boxes": boxes,
            }
        )

    urls = [q.db.urls[int(dbidx)].replace("thumbnails/", "") for dbidx in next_idxs]
    pdata = {
        "image_urls": urls,
        "ldata": reslabs,
    }
    return pdata


def auto_fill_boxes(ground_truth_df, ldata):
    """simulates human feedback by generating ldata from ground truth data frame"""
    new_ldata = copy.deepcopy(ldata)
    for rec in new_ldata:
        rows = ground_truth_df[ground_truth_df.dbidx == rec["dbidx"]]
        rows = rows.rename(
            mapper={"x1": "xmin", "x2": "xmax", "y1": "ymin", "y2": "ymax"}, axis=1
        )
        rows = rows[["xmin", "xmax", "ymin", "ymax"]]
        rec["boxes"] = rows.to_dict(orient="records")
    return new_ldata


def update_db(label_db, ldata):
    for rec in ldata:
        r = rec["boxes"]
        label_db[rec["dbidx"]] = r


# def make_image_panel(bfq, idxbatch):
#     # from .ui import widgets

#     dat = get_panel_data(bfq, bfq.label_db, idxbatch)

#     ldata = dat['ldata']
#     if bfq.auto_fill_df is not None:
#         gt_ldata = auto_fill_boxes(bfq.auto_fill_df, ldata)
#         ## only use boxes for things we have not added ourselves...
#         ## (ie don't overwrite db)
#         for rdb, rref in zip(ldata, gt_ldata):
#             if rdb['dbidx'] not in bfq.label_db:
#                 rdb['boxes'] = rref['boxes']

#     pn = widgets.MImageGallery(**dat)
#     return pn


def update_rois(bfq, ldata, pad_factor, augment_n):
    rois = extract_rois(bfq.db, ldata, pad_factor)
    bfq.rois.append(rois)

    if augment_n > 0:
        arois = augment(rois, n=augment_n)
    else:
        arois = rois
    bfq.augmented_rois.append(arois)

    roivecs = embed_rois(bfq.db, arois)
    if roivecs.shape[0] > 0:
        nrois = sklearn.preprocessing.normalize(roivecs)
    else:
        nrois = roivecs
    bfq.roi_vecs.append(nrois)

    # normalize bc. we use cosine sim, but also
    # want to average etc...
    assert len(bfq.roi_vecs) == len(bfq.rois)
    assert len(bfq.rois) == len(bfq.augmented_rois)
    assert len(bfq.rois) == len(bfq.acc_idxs) + 1


from tqdm.auto import tqdm


def binary_panel_data(ldata):
    dbidx = np.array([d["dbidx"] for d in ldata])
    pred = np.array([len(d["boxes"]) > 0 for d in ldata])
    return dbidx, pred
