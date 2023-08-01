from ray.data.extensions import TensorArray
import numpy as np
import pandas as pd
from seesaw.dataset_tools import *
from tqdm.auto import tqdm
from seesaw.labeldb import LabelDB
from seesaw.query_interface import  AccessMethod, InteractiveQuery

from seesaw.models.embeddings import make_clip_transform, ImTransform, XEmbedding
import pyroaring as pr
from seesaw.vector_index import VectorIndex
from seesaw.definitions import resolve_path
import os


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


from seesaw.box_utils import left_iou_join

def match_labels_to_vectors(label_db: LabelDB, vec_meta: pd.DataFrame, target_description=None):
    ''' given a set of box labels, and a vector index with box info, 
        for each vector in an image, find the maximum label overlap with it
        and use that as a score.

    '''
    idxs = label_db.get_seen()
    vec_meta = vec_meta[vec_meta.dbidx.isin(idxs)]
    boxdf = label_db.get_box_df(return_description=True)
    #print(f'{boxdf=}')

    if target_description is not None:
        target_df = boxdf[boxdf.description == target_description]
    else:
        target_df = boxdf[boxdf.marked_accepted > 0]
    vec_meta_new = left_iou_join(vec_meta, target_df)

    vec_meta_new = vec_meta_new.assign(ys = (vec_meta_new.max_iou > 0).astype('float'))
    return vec_meta_new


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


def simple_score_frame(meta_df, rescore_fun):
    pass


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
        excluded : pr.BitMap = None
    ):
        self.embedding = embedding
        self.path = path
        self.excluded = pr.BitMap([]) if excluded is None else excluded

        if min_zoom_level == 1:
            self.vectors = vectors
            self.vector_meta = vector_meta
            self.vec_index = vec_index
            self.all_indices = pr.FrozenBitMap(self.vector_meta.dbidx.values) - self.excluded
        else:  # filter out lowest zoom level
            print("WARNING: filtering out min_zoom_level")
            mask = filter_mask(vector_meta, min_level_inclusive=min_zoom_level)
            self.vector_meta = vector_meta[mask].reset_index(drop=True)
            self.vectors = vectors[mask]

            self.vec_index = None  # no index constructed here
            self.all_indices = pr.FrozenBitMap(self.vector_meta.dbidx.values) - self.excluded
        

    @staticmethod
    def from_path(index_path: str, *, use_vec_index=True, **options):
        from ...services import get_parquet, get_model_actor
        print(f'{__file__}:{options=}')
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
            path = index_path,
            excluded=options.get('excluded', None)
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
        vector2=None,
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

        if vector2 is not None:
            scores2 = vectors @ vector2.reshape(-1)
            scores = scores - scores2

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
        assert self.index is not None

    def getXy(self, get_positions=False, target_description=None):
        matched_df = match_labels_to_vectors(self.label_db, self.index.vector_meta, target_description=target_description)
    
        if get_positions:
            pos = matched_df.index[matched_df.ys > 0].values
            neg = matched_df.index[matched_df.ys == 0].values
            return pos, neg
        else:
            return matched_df[['dbidx', 'ys', 'max_iou']]