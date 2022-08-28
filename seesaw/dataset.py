from .dataset_tools import ExplicitPathDataset
from .definitions import resolve_path
import os
import pandas as pd
import numpy as np
import math
import glob
import json


def list_image_paths(basedir, prefixes=[""], extensions=["jpg", "jpeg", "png"]):
    acc = []
    for prefix in prefixes:
        for ext in extensions:
            pattern = f"{basedir}/{prefix}/**/*.{ext}"
            imgs = glob.glob(pattern, recursive=True)
            acc.extend(imgs)
            print(f"found {len(imgs)} files with pattern {pattern}...")

    relative_paths = [f[len(basedir) :].lstrip("./") for f in acc]
    return list(set(relative_paths))


def infer_qgt_from_boxes(box_data, num_files):
    qgt = box_data.groupby(["dbidx", "category"]).size().unstack(level=1).fillna(0)
    qgt = qgt.reindex(np.arange(num_files)).fillna(0)
    return qgt.clip(0, 1)


def prep_ground_truth(paths, box_data, qgt):
    """adds dbidx column to box data, sets dbidx in qgt and sorts qgt by dbidx"""
    path2idx = dict(zip(paths, range(len(paths))))
    mapfun = lambda x: path2idx.get(x, -1)
    box_data = box_data.assign(dbidx=box_data.file_path.map(mapfun).astype("int"))
    box_data = box_data[box_data.dbidx >= 0].reset_index(drop=True)

    new_ids = qgt.index.map(mapfun)
    qgt = qgt[new_ids >= 0]
    qgt = qgt.set_index(new_ids[new_ids >= 0])
    qgt = qgt.sort_index()

    ## Add entries for files with no labels...
    qgt = qgt.reindex(np.arange(len(paths)))  # na values will be ignored...

    assert len(paths) == qgt.shape[0], "every path should be in the ground truth"
    return box_data, qgt

import ray.data
from ray.data.datasource.file_meta_provider import FastFileMetadataProvider

def get_default_qgt(dataset, box_data):
    """ """
    import scipy.sparse
    row_ind = box_data.dbidx.values
    col_ind = box_data.category.values.codes
    values = np.ones_like(row_ind)

    height = dataset.file_meta.shape[0]
    width =  box_data.category.dtype.categories.shape[0]
    mat = scipy.sparse.csc_matrix((values, (row_ind, col_ind)),  shape = (height, width))
    
    qgt = pd.DataFrame.sparse.from_spmatrix(mat, index=dataset.file_meta.index.values, columns=box_data.category.dtype.categories)
    return qgt


class SeesawDatasetManager:
    def __init__(self, dataset_path, cache=None):
        """Assumes layout created by create_dataset"""
        dataset_path = resolve_path(dataset_path)
        self.dataset_name = os.path.basename(dataset_path)
        self.dataset_root = dataset_path
        file_meta = pd.read_parquet(f"{self.dataset_root}/file_meta.parquet")
        self.file_meta = file_meta
        self.paths = file_meta["file_path"].values
        self.image_root = os.path.realpath(f"{self.dataset_root}/images/")
        self.cache = cache

    def get_pytorch_dataset(self):
        return ExplicitPathDataset(
            root_dir=self.image_root, relative_path_list=self.paths
        )
    
    def get_url(self, dbidx, host='localhost.localdomain:10000') -> str:
        return f"http://{host}/{self.image_root}/{self.paths[dbidx]}"

    def as_ray_dataset(ds, limit=None, parallelism=-1) -> ray.data.Dataset:
        """ with schema {dbidx: int64, binary: object}
        """
        path_array = (ds.image_root + '/') + ds.paths
        path_array = path_array[:limit]
        read_paths = list(path_array)
        
        def fix_batch(batch):
            inv_paths = {p:dbidx for dbidx,p in enumerate(read_paths)}

            def tup_fixer(tup):
                tup = tup[0]
                path, binary = tup
                dbidx = inv_paths[path]
                return (dbidx, binary)

            return batch.apply(tup_fixer, axis=1, result_type='expand').rename({0:'dbidx', 1:'binary'}, axis=1)
    
        binaries = ray.data.read_binary_files(paths=read_paths, include_paths=True, 
                    parallelism=parallelism, meta_provider=FastFileMetadataProvider())
    
        return binaries.lazy().map_batches(fix_batch, batch_format='pandas')

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dataset_name})"

    def save_ground_truth(self, box_data, qgt=None):
        """
        Will add qgt and box information. or overwrite it.
        """
        if qgt is None:
            qgt = infer_qgt_from_boxes(box_data, num_files=self.paths.shape[0])

        assert qgt.shape[0] == self.paths.shape[0]
        gt_root = self.ground_truth_path()
        os.makedirs(gt_root, exist_ok=True)
        box_data.to_parquet(f"{gt_root}/boxes.parquet")
        qgt.to_parquet(f"{gt_root}/qgt.parquet")

    def ground_truth_path(self):
        gt_root = f"{self.dataset_root}/ground_truth/"
        return gt_root

    def load_eval_categories(self):
        assert os.path.exists(f"{self.dataset_root}/ground_truth")
        return json.load(open(f"{self.dataset_root}/ground_truth/categories.json"))

    def load_ground_truth(self):
        assert os.path.exists(f"{self.dataset_root}/ground_truth")
            
        box_data = self.cache.read_parquet(
            f"{self.dataset_root}/ground_truth/box_data.parquet"
        )

        qgt_path = f"{self.dataset_root}/ground_truth/qgt.parquet"
        if os.path.exists(qgt_path):
            qgt = self.cache.read_parquet(qgt_path)
            box_data, qgt = prep_ground_truth(self.paths, box_data, qgt)
            return box_data, qgt
        else: ## note: this will be wrong for subsets. TODO fix this.
            return box_data, get_default_qgt(self, box_data)

    ## TODO: add subset method that takes care of ground truth, and URLS, and that 
    ### can be used by indices over the subset?

def create_dataset(image_src, output_path, paths=[]) -> SeesawDatasetManager:
    """
    if not given explicit paths, it assumes every jpg, jpeg and png is wanted
    """
    assert not os.path.exists(output_path), "output already exists"
    dirname = os.path.dirname(output_path)
    os.makedirs(dirname, exist_ok=True)
    basename = os.path.basename(output_path)

    final_dspath = f"{dirname}/{basename}"
    dspath = f"{dirname}/.tmp.{basename}"

    assert not os.path.exists(final_dspath), "name already used"
    if os.path.exists(dspath):
        os.rmdir(dspath)
    image_src = resolve_path(image_src)
    assert os.path.isdir(image_src)

    os.mkdir(dspath)
    image_path = f"{dspath}/images"
    os.symlink(image_src, image_path)
    if len(paths) == 0:
        paths = list_image_paths(image_src)
        paths = sorted(paths)

    df = pd.DataFrame({"file_path": paths})
    df.to_parquet(f"{dspath}/file_meta.parquet")

    os.rename(dspath, final_dspath)
    return SeesawDatasetManager(final_dspath)
