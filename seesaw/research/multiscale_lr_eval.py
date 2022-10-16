from seesaw.seesaw_session import get_subset
from seesaw.query_interface import LabelDB
from seesaw.indices.multiscale.multiscale_index import get_pos_negs_all_v2
from seesaw.seesaw_bench import fill_imdata

from sklearn.metrics import average_precision_score, roc_auc_score, ndcg_score
import numpy as np
import pandas as pd
from seesaw.basic_types import Box
from sklearn.model_selection import train_test_split

from seesaw.dataset_search_terms import category2query
from seesaw.logistic_regression import LogisticRegresionPT


def get_metrics(qstr, vec, df):
    results = []
    Xs = df.vectors.to_numpy()
    
    if isinstance(vec,np.ndarray):
        scores = Xs @ vec.reshape(-1)
    else:
        scores = vec.predict_proba(Xs).reshape(-1)

    ys = df['ys'].values

    ap = average_precision_score(ys, scores)
    roc_auc = roc_auc_score(ys, scores)
    ndcg = ndcg_score(np.array([ys.astype('float')]), np.array([scores]))
    ndcg30 = ndcg_score(np.array([ys.astype('float')]), np.array([scores]), k=30)

    orders = np.argsort(-scores)
    a = np.nonzero(ys[orders])[0]

    results.append({'qstr':qstr, 'npos':ys.sum(), 'ap':ap, 'ndcg':ndcg, 
                    'rank_first':a[0],
                    'rank_second':a[1],
                    'rank_third':a[2],
                    'rank_tenth':a[9],
                    'ndcg30':ndcg30, 'roc_auc':roc_auc})
        
    return pd.DataFrame(results)

from ray.data.extensions import TensorArray

def eval_multiscale_lr(objds, idx, category):
    subidx, boxes, present, positive = get_subset(objds, idx, category)
    bx = boxes[boxes.category == category]
    
    ldb = LabelDB()
    for dbidx, gp in bx.groupby('dbidx'):
        bxlist = [Box(**b) for b in gp.to_dict(orient="records")]
        ldb.put(dbidx, bxlist)

    for dbidx in set(present) - set(positive):
        ldb.put(dbidx, [])
        
    dbidxs = subidx.vector_meta.dbidx.unique()
    pos, neg = get_pos_negs_all_v2(dbidxs, ldb, vec_meta=subidx.vector_meta)
    
    indices = np.concatenate([pos, neg])
    ys = np.concatenate([np.ones(np.array(pos).shape[0]), np.zeros(np.array(neg).shape[0])])
    vec_meta = subidx.vector_meta.iloc[indices]
    vecs = subidx.vectors[indices]
    vec_meta = vec_meta.assign(vectors=TensorArray(vecs), ys=ys)
    
    image_labels = vec_meta.groupby('dbidx').ys.max().reset_index()
    tr_idcs, tst_idcs = train_test_split(image_labels['dbidx'].values, stratify=image_labels['ys'].values)
    train_meta = vec_meta[vec_meta.dbidx.isin(set(tr_idcs))]
    test_meta = vec_meta[vec_meta.dbidx.isin(set(tst_idcs))]
    
    qstr = category2query(dataset='lvis', cat=category)
    reg_vec = idx.string2vec(qstr)
    lr = LogisticRegresionPT(class_weights=1., scale='centered', reg_lambda = 1., verbose=True, fit_intercept=False, 
                         regularizer_vector=reg_vec)
    
    lr.fit(train_meta.vectors.to_numpy(), train_meta.ys.values.reshape(-1,1))
    
    m_train = get_metrics(category, reg_vec, train_meta).assign(split='train')
    m_val = get_metrics(category, reg_vec, test_meta).assign(split='test')
    return pd.concat([m_train, m_val], ignore_index=True)