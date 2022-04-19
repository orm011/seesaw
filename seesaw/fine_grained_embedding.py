import numpy as np
import pyroaring as pr
import os
import math

from .search_loop_models import *
from .embeddings import *

## want function that selects by dbidx and then rewrites them to be sequential.


def restrict_fine_grained(vec_meta, vec, indxs):
    assert vec_meta.shape[0] == vec.shape[0]
    assert (indxs[1:] > indxs[:-1]).all(), "must be sorted"
    mask = vec_meta.dbidx.isin(pr.BitMap(indxs))
    if mask.all():
        return vec_meta, vec

    vec_meta = vec_meta[mask]
    vec = vec[mask]
    lookup_table = np.zeros(vec_meta.dbidx.max() + 1).astype("int") - 1
    lookup_table[indxs] = np.arange(indxs.shape[0], dtype="int")
    new_dbidx = lookup_table[vec_meta.dbidx]
    assert (new_dbidx >= 0).all()
    vec_meta = vec_meta.assign(dbidx=new_dbidx)  # this line shows up in profiler
    assert (
        vec_meta.dbidx.unique().shape[0] == indxs.shape[0]
    ), "missing fine-grained embedding for some of the indices requested"
    assert vec_meta.shape[0] == vec.shape[0]
    return vec_meta.reset_index(drop=True), vec


def resize_to_grid(base_size):
    """
    makes a transform that increases size and crops an image and any boxes so that the output meets the following
        1. size of each side is least equal to base size
        2. base size fits an integer number of times along both dimensions using a stride of base_size//2
            the shorter side gets magnified, and the longer side gets cropped
        3. only tested with base_size = 224, what clip requires
        4. boxes assocated with image are also transformed to match the new image
        5. each image is transformed differently dependig on its dimensions
    """
    hsize = base_size // 2

    def fun(*, im=None, boxes=None):
        if im is None and boxes.shape[0] == 0:
            return None, boxes
        elif im is None:
            (w, h) = boxes.im_width.iloc[0], boxes.im_height.iloc[0]
        else:
            (w, h) = im.size

        sz = min(w, h)

        ## edge case:  side < 224 => side == 224.
        # max(2,math.ceil(sz/hsize))
        round_up = math.ceil(sz / hsize) * hsize
        scale_factor = max(base_size, round_up) / sz
        target_h = int(math.ceil(scale_factor * h))
        target_w = int(math.ceil(scale_factor * w))
        assert target_h >= base_size
        assert target_w >= base_size

        htrim = target_h % hsize
        wtrim = target_w % hsize
        crop_h = target_h - htrim
        crop_w = target_w - wtrim

        tx = T.Compose(
            [
                T.Resize((target_h, target_w), interpolation=PIL.Image.BICUBIC),
                T.CenterCrop((crop_h, crop_w)),
            ]
        )

        assert crop_h >= base_size
        assert crop_w >= base_size
        assert crop_h % hsize == 0
        assert crop_w % hsize == 0

        if boxes is not None:
            box0 = boxes
            sf = scale_factor
            txbox = box0.assign(
                x1=np.clip(box0.x1 * sf - wtrim / 2, 0, crop_w),
                x2=np.clip(box0.x2 * sf - wtrim / 2, 0, crop_w),
                y1=np.clip(box0.y1 * sf - htrim / 2, 0, crop_h),
                y2=np.clip(box0.y2 * sf - htrim / 2, 0, crop_h),
                im_width=crop_w,
                im_height=crop_h,
            )
        else:
            txbox = None

        if im is not None:
            txim = tx(im)
        else:
            txim = None

        return txim, txbox

    return fun


def nearest_j(midx, width, base_size, mode="nearest"):
    hsize = base_size // 2
    nlines = width // hsize - 1

    midx_scaled = midx / hsize

    if mode == "nearest":
        ln = np.round(midx_scaled) - 1
    elif mode == "start":
        ln = np.floor(midx_scaled) - 1
    elif mode == "end":
        ln = np.floor(midx_scaled)

    line_id = np.clip(ln, 0, nlines - 1).astype("int")
    return line_id

    ## sanity: if
    # midx_scaled = 0 or 1, should map to j = 0
    # midx_scaled = 2 => j = 1
    # midx_scaled = width/size or width/size + 1 => width//size -1

    # the center of each window is located at
    # np.arange(1,nlines+1)*hsize
    ## draw diagram to see why we subtract.
    # midx = midx - hsize/2
    # return np.clip(np.round(midx/hsize), 0, nlines).astype('int')
    # midx = np.clip(midx, a_min=0, a_max=width)
    # next_line_idx = midx//hsize
    # next_line_idx = np.clip(next_line_idx, a_min=0,a_max=nlines-1)
    # return next_line_idx.astype('int')


def nearest_ij(box, base_size):
    xs = (box.x1 + box.x2) / 2
    jjs = nearest_j(xs, box.im_width, base_size=224, mode="nearest").values

    ys = (box.y1 + box.y2) / 2
    iis = nearest_j(ys, box.im_height, base_size=224, mode="nearest").values
    return pd.DataFrame({"i": iis, "j": jjs}, index=box.index)


def x2i(start, end, total, base_size=224):
    ## return i,j s for squares with that position...
    # hsize = base_size//2
    # max_i = (total - hsize)//hsize - 1 # stride of base_size//2, last stride doesn't count.
    # i1 = np.clip(start//hsize-1,0,max_i).astype('int') # easy case
    # i2 = np.clip(end//hsize,0,max_i).astype('int')
    i1 = nearest_j(start, total, base_size=224, mode="start").values
    i2 = nearest_j(end, total, base_size=224, mode="end").values

    # this is correct except for the last one: in that case it should be the previous one...
    return i1, i2 + 1


def box2ij(box, base_size):
    i1, i2 = x2i(start=box.y1, end=box.y2, total=box.im_height, base_size=base_size)
    j1, j2 = x2i(start=box.x1, end=box.x2, total=box.im_width, base_size=base_size)
    return pd.DataFrame({"i1": i1, "i2": i2, "j1": j1, "j2": j2})


def crop(im, i, j):
    return im.crop((j * 112, i * 112, j * 112 + 224, i * 112 + 224))


def get_pos_negs(box, vec_meta):
    """
    For a given image im, and a list of boxes (dataframe)
    and metadata of image vectors, compute
     1. vectors of image chunks that do not overlap at all
     2. vectors of chunks nearest to box center.
    """
    if box.shape[0] == 0:
        neg_idxs = pr.BitMap(vec_meta.index.values)
        pos_idxs = pr.BitMap()
        return pos_idxs, neg_idxs

    ijs = box2ij(box, base_size=224)
    nearest_ijs = nearest_ij(box, base_size=224)
    tmp_meta = vec_meta
    negatives = []
    centers = []
    for tup, ctup in zip(ijs.itertuples(), nearest_ijs.itertuples()):
        overlap_ijs = tmp_meta.iis.between(tup.i1, tup.i2 - 1) & (
            tmp_meta.jjs.between(tup.j1, tup.j2 - 1)
        )
        negs = tmp_meta[~overlap_ijs]  # no overlap whatsoever
        negatives.append(pr.BitMap(negs.index))
        cent = tmp_meta[(tmp_meta.iis == ctup.i) & (tmp_meta.jjs == ctup.j)]
        centers.append(pr.BitMap(cent.index))

    neg_idxs = pr.BitMap.intersection(*negatives)
    pos_idxs = pr.BitMap.union(*centers)
    return pos_idxs, neg_idxs


def get_pos_negs_all(dbidxs, ds, vec_meta):
    pos = []
    neg = []
    for idx in dbidxs:
        box = ds[idx]
        p, n = get_pos_negs(box, vec_meta[vec_meta.dbidx == idx])
        pos.append(p)
        neg.append(n)

    posidxs = pr.BitMap.union(*pos)
    negidxs = pr.BitMap.union(*neg)
    return posidxs, negidxs
