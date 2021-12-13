from .embeddings import *
from .cross_modal_db import EmbeddingDB
import json
import pandas as pd
import os
from .fine_grained_embedding import *
import inspect, os, copy
from .dataset_tools import ExplicitPathDataset

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)[:,np.newaxis]
    return vecs/(norms + 1e-6)

class EvDataset(object):
    query_ground_truth : pd.DataFrame
    box_data : pd.DataFrame
    embedding : XEmbedding
    embedded_dataset : np.ndarray
    fine_grained_embedding : np.ndarray
    fine_grained_meta : pd.DataFrame
    imge_dataset : ExplicitPathDataset
    vec_index_path : str

    def __init__(self, *, root, paths, embedded_dataset, query_ground_truth, box_data, embedding, 
                fine_grained_embedding=None, fine_grained_meta=None, vec_index_path=None, vec_index=None):
        """
        meant to be cheap to run, we use it to make subsets.
            do any global changes outside, eg in make_evdataset.
        """
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        kwargs = {k:v for (k,v) in values.items() if k in args}
        self.__dict__.update(kwargs)
        self.image_dataset = ExplicitPathDataset(root_dir=root, relative_path_list=paths)
        self.vec_index = vec_index
        
    def __len__(self):
        return len(self.image_dataset)

def get_class_ev(ev, category, boxes=False): 
    """restrict dataset to only those indices labelled for a given category"""
    gt = ev.query_ground_truth[category]
    class_idxs = gt[~gt.isna()].index.values
    ev1 = extract_subset(ev, idxsample=class_idxs, categories=[category],boxes=boxes)
    return ev1, class_idxs

def extract_subset(ev : EvDataset, *, idxsample=None, categories=None, boxes=True) -> EvDataset:
    """makes an evdataset consisting of those categories and those index samples only
    """
    if categories is None:
        categories = ev.query_ground_truth.columns.values

    if idxsample is None:
        idxsample = np.arange(len(ev), dtype='int')
    else:
        assert (np.sort(idxsample) == idxsample).all(), 'sort it because some saved data assumed it was sorted here'

    categories = set(categories)
    idxset = pr.BitMap(idxsample)

    qgt = ev.query_ground_truth[categories]
    embedded_dataset = ev.embedded_dataset
    paths = ev.paths
    vec_index = ev.vec_index

    if not (idxset == pr.BitMap(range(len(ev)))): # special case where everything is copied
        print('warning: subset operation forces foregoing pre-built vector index.')
        qgt = qgt.iloc[idxsample].reset_index(drop=True)
        embedded_dataset = embedded_dataset[idxsample]
        paths = ev.paths[idxsample]
        vec_index = None

    if boxes:
        good_boxes = ev.box_data.dbidx.isin(idxset) & ev.box_data.category.isin(categories)
        if good_boxes.all(): # avoid copying when we want everything anyway (common case)
            sample_box_data = ev.box_data
        else:
            sample_box_data = ev.box_data[good_boxes]
            lookup_table = np.zeros(len(ev), dtype=np.int) - 1
            lookup_table[idxsample] = np.arange(idxsample.shape[0], dtype=np.int)
            sample_box_data = sample_box_data.assign(dbidx = lookup_table[sample_box_data.dbidx])
            assert (sample_box_data.dbidx >= 0 ).all()
    else:
        sample_box_data = None

    if ev.fine_grained_embedding is not None:
        meta, vec = restrict_fine_grained(ev.fine_grained_meta,ev.fine_grained_embedding, idxsample)
    else:
        meta = None
        vec = None

    return EvDataset(root=ev.root, paths=paths,
                    embedded_dataset=embedded_dataset,
                    query_ground_truth=qgt,
                    box_data=sample_box_data,
                    embedding=ev.embedding,
                    fine_grained_embedding=vec,
                    fine_grained_meta=meta, vec_index=vec_index)