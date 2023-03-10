#from seesaw.seesaw_session import get_subset
from seesaw.query_interface import LabelDB

from sklearn.metrics import average_precision_score, roc_auc_score, ndcg_score
import numpy as np
import pandas as pd
from seesaw.basic_types import Box
from sklearn.model_selection import train_test_split

from seesaw.dataset_search_terms import category2query
from seesaw.logistic_regression import LogisticRegressionPT


def get_scores(vec, df):
    Xs = df.vectors.to_numpy()
    
    if isinstance(vec,np.ndarray):
        scores = Xs @ vec.reshape(-1)
    elif hasattr(vec, 'predict_proba'):
        scores = vec.predict_proba(Xs).reshape(-1)
    else:
        assert False

    return scores

def get_metrics(df, scores, frame_pooling):
    df = df.assign(scores=scores)
    results = []
    ys  = df['ys'].values

    if frame_pooling:
        aggdf = df.groupby('dbidx')[['scores', 'ys']].max()
        scores = aggdf['scores'].values
        ys = aggdf['ys'].values

    ap = average_precision_score(ys, scores)
#    roc_auc = roc_auc_score(ys, scores)
#    ndcg = ndcg_score(np.array([ys.astype('float')]), np.array([scores]))
#    ndcg30 = ndcg_score(np.array([ys.astype('float')]), np.array([scores]), k=30)

    orders = np.argsort(-scores)
    a = np.nonzero(ys[orders])[0]
    npos  = ys.sum()
    results.append({'ntotal':ys.shape[0], 'npos':npos, 'ap':ap, 
    #'ndcg':ndcg, 
                    'rank_first':a[0] if npos > 0 else np.nan,
                    'rank_second':a[1] if npos > 1 else np.nan,
                    'rank_third':a[2] if npos > 2 else np.nan,
                    'rank_tenth':a[9] if npos > 10 else np.nan,
                    'frame_pooling':frame_pooling
#                    'ndcg30':ndcg30, 'roc_auc':roc_auc
                    })
        
    return pd.DataFrame(results)

from ray.data.extensions import TensorArray
import seesaw.indices.coarse
import seesaw.dataset_manager
from seesaw.dataset_manager import GlobalDataManager
from seesaw.box_utils import left_iou_join

def train_test_split_framewise(vec_meta, random_state):
    image_labels = vec_meta.groupby('dbidx').ys.max().reset_index()
    tr_idcs, tst_idcs = train_test_split(image_labels['dbidx'].values, stratify=image_labels['ys'].values, random_state=random_state)
    train_meta = vec_meta[vec_meta.dbidx.isin(set(tr_idcs))]
    test_meta = vec_meta[vec_meta.dbidx.isin(set(tst_idcs))]
    return train_meta, test_meta

def eval_multiscale_lr(ds, *, idx, boxes, category, frame_pooling, random_state, RegressionClass, fit_only, **   lropts):
    if ds.path.find('lvis') >= 0:
        ds = ds.load_subset(category)

    # idx = ds.load_index(idxname,  options=dict(use_vec_index=False))
    # boxes, _ = ds.load_ground_truth()
    boxes = boxes[boxes.category == category]

    vec_meta = left_iou_join(idx.vector_meta, boxes)
    vec_meta = vec_meta.assign(vectors=TensorArray(idx.vectors), ys=(vec_meta.max_iou > 0).astype('float32'))

    if fit_only:
        train_meta = vec_meta
        test_meta = None
    else:
        train_meta, test_meta = train_test_split_framewise(vec_meta, random_state)

    # if change_score_scale:
    #     pseudo_label, fully_labeled = train_test_split_framewise(train_meta, random_state+1)

    #     ys = pseudo_label.ys.values
    #     ys = ys/10. + .3
    #     pseudo_label = pseudo_label.assign(ys=ys)

    #     train_meta = pd.concat([pseudo_label, fully_labeled], ignore_index=True)

    lr = RegressionClass(**lropts)
    lr.fit(train_meta.vectors.to_numpy(), train_meta.ys.values.reshape(-1,1).astype('float'))

    scores_train = get_scores(lr, train_meta)
    train_metrics =  get_metrics(train_meta, scores=scores_train, frame_pooling=frame_pooling).assign(split='train', method='lr')

    if not fit_only:
        scores_test = get_scores(lr, test_meta)
        test_metrics = [get_metrics(test_meta, scores=scores_test, frame_pooling=frame_pooling).assign(split='test', method='lr')]
    else:
        test_metrics = []

    dfs2 = [train_metrics] + test_metrics
    return pd.concat(dfs2, ignore_index=True).assign(category=category)