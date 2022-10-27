#from seesaw.seesaw_session import get_subset
from seesaw.query_interface import LabelDB
from seesaw.indices.multiscale.multiscale_index import get_pos_negs_all_v2
from seesaw.seesaw_bench import fill_imdata

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

def get_metrics(df, ys, scores, frame_pooling):
    df = df.assign(scores=scores, ys=ys)
    results = []
    # ys  = df['ys'].values

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
#                    'ndcg30':ndcg30, 'roc_auc':roc_auc
                    })
        
    return pd.DataFrame(results)

from ray.data.extensions import TensorArray
import seesaw.indices.coarse
import seesaw.dataset_manager
from seesaw.dataset_manager import GlobalDataManager
from seesaw.box_utils import left_iou_join

def train_test_split_framewise(vec_meta):
    image_labels = vec_meta.groupby('dbidx').ys.max().reset_index()
    tr_idcs, tst_idcs = train_test_split(image_labels['dbidx'].values, stratify=image_labels['ys'].values)
    train_meta = vec_meta[vec_meta.dbidx.isin(set(tr_idcs))]
    test_meta = vec_meta[vec_meta.dbidx.isin(set(tst_idcs))]
    return train_meta, test_meta

def eval_multiscale_lr(root, idxname, category):
    gdm = GlobalDataManager(root)
    ds = gdm.get_dataset('lvis').load_subset(category)
    idx = ds.load_index(idxname,  options=dict(use_vec_index=False))

    boxes, _ = ds.load_ground_truth()

    vec_meta = left_iou_join(idx.vector_meta, boxes)
    vec_meta = vec_meta.assign(vectors=TensorArray(idx.vectors), ys=vec_meta.max_iou > 0)    
    train_meta, test_meta = train_test_split_framewise(vec_meta)
    
    lr = LogisticRegressionPT(class_weights='balanced', scale='centered', reg_lambda = 10., verbose=True, fit_intercept=False, 
                         regularizer_vector='norm')
    
    lr.fit(train_meta.vectors.to_numpy(), train_meta.ys.values.reshape(-1,1).astype('float'))    
    dfs2 = [
        get_metrics(lr, train_meta, frame_pooling=True).assign(split='train', method='lr'),  
        get_metrics(lr, test_meta, frame_pooling=True).assign(split='test', method='lr')
    ]
    return pd.concat(dfs2, ignore_index=True).assign(category=category)