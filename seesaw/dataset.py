from .dataset_tools import ExplicitPathDataset
from .definitions import resolve_path
import os
import pandas as pd
import numpy as np
import math
import glob
import json


def list_image_paths(basedir, prefixes=[""], extensions=["jpg", "jpeg", "png", 'tif', 'tiff']):
    acc = []
    extensions = set(extensions)
    for prefix in prefixes:
        pattern = f"{basedir}/{prefix}/**/*.*"
        files = glob.glob(pattern, recursive=True)
        imgs = [f for f in files if f.split(".")[-1].lower() in extensions]
        acc.extend(imgs)
        print(f"found {len(imgs)} files with pattern in {extensions}...")

    relative_paths = [f[len(basedir) :].lstrip("./") for f in acc]
    return sorted(list(set(relative_paths)))


def infer_qgt_from_boxes(box_data, num_files):
    qgt = box_data.groupby(["dbidx", "category"]).size().unstack(level=1).fillna(0)
    qgt = qgt.reindex(np.arange(num_files)).fillna(0)
    return qgt.clip(0, 1)


def fix_dbidx(box_data, paths):
    path2idx = dict(zip(paths, range(len(paths))))
    mapfun = lambda x: path2idx.get(x, -1)
    box_data = box_data.assign(dbidx=box_data.file_path.map(mapfun).astype("int"))
    box_data = box_data[box_data.dbidx >= 0].reset_index(drop=True)
    return box_data

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

from .query_interface import AccessMethod


class BaseDataset: # common interface for datasets and their subsets
    image_root : str
    file_meta : pd.DataFrame # indexed by dbidx, column 'file_path'
    def load_index(self, index_name, *, options):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError
    
    def get_image_paths(self, dbidxs):
        paths = []
        for dbidx in dbidxs:
            subpath = self.file_meta.loc[dbidx]
            path = f'{self.image_root}/{subpath}'
            paths.append(path)
        # urls = get_image_paths(self.dataset.image_root, self.dataset.paths, idxbatch)
        return paths

    def load_subset(self, subset_name):
        raise NotImplementedError
    
    def leave_one_out(self, dbidx):
        return LeaveOneOutSubset(self, excluded_dbidx=dbidx)

    def get_url(self, dbidx, host) -> str:
        raise NotImplementedError

    def get_urls(self, dbidxs, host='/'):
        urls = []
        for dbidx in dbidxs:
            url = self.get_url(dbidx, host=host)
            urls.append(url)
        return urls

    def as_ray_dataset(self, limit=None, parallelism=-1) -> ray.data.Dataset:
        raise NotImplementedError

    def load_eval_categories(self):
        raise NotImplementedError

    def load_ground_truth(self):
        raise NotImplementedError

from .services import get_parquet

def process_annotation_session(annotation_session_path, image_root):
    image_root = os.path.normpath(image_root) + '/'
    resd = json.load(open(annotation_session_path + '/summary.json', 'r'))
    category = resd['session']['params']['annotation_category']
    records = []
    seen_paths = set()
    for row in resd['session']['gdata']:
        for imdata in row:
            path = imdata['url'][len(image_root):]
            assert path not in seen_paths
            seen_paths.add(path)
            if imdata['boxes'] is None:
                continue
            for rec in imdata['boxes']:
                if rec['marked_accepted']:
                    rec['file_path'] = path
                    rec['category'] = category
                    del rec['description']
                    del rec['marked_accepted']
                    records.append(rec)
                    
    return category, seen_paths, pd.DataFrame.from_records(records)

def ammend_annotations(image_root, box_data, annotation_session_path):
    bd = box_data
    category, seen_paths, amended_boxes = process_annotation_session(annotation_session_path, image_root)
    ammend_mask = (bd.category == category) & (bd.file_path.isin(seen_paths))
    bd = bd[~ammend_mask]
    amended_boxes = amended_boxes.assign(origin=annotation_session_path)
    return pd.concat([bd, amended_boxes], ignore_index=True)

def get_subdirs_chronological(annpath):
    paths = os.listdir(annpath)
    subdirs = []
    mtimes = []
    for s in paths:
        fpath = f'{annpath}/{s}'
        if os.path.isdir(fpath):
            mtimes.append(os.stat(fpath).st_mtime)
            subdirs.append(fpath)

    pairs = zip(subdirs, mtimes)
    ordered_subdirs = [a for (a,b) in sorted(pairs, key=lambda p : p[1])]
    return ordered_subdirs

class SeesawDataset(BaseDataset):
    def __init__(self, dataset_path):
        """Assumes layout created by create_dataset"""
        dataset_path = resolve_path(dataset_path)
        self.path = dataset_path
        self.dataset_name = os.path.basename(dataset_path)
        self.dataset_root = dataset_path
        file_meta = pd.read_parquet(f"{self.dataset_root}/file_meta.parquet")
        self.file_meta = file_meta
        self.paths = file_meta.file_path.values
        self.dbidx_map = dict(zip(self.paths, self.file_meta.index.values)) # maps str
        self.image_root = os.path.realpath(f"{self.dataset_root}/images/")

    @staticmethod
    def create_from_directory(dataset_path, image_dir, force=False) -> "SeesawDataset":
        return create_dataset(image_dir, dataset_path, paths=[], force=force)

    def size(self):
        return self.file_meta.shape[0]

    def get_pytorch_dataset(self):
        return ExplicitPathDataset(
            root_dir=self.image_root, relative_path_list=self.paths
        )

    def list_indices(self):
        return os.listdir(f'{self.path}/indices/')

    def load_index(self, index_name, *, options):
        index_path = f"{self.path}/indices/{index_name}"
        return AccessMethod.load(index_path, options=options)

    def _create_subset(self, subset_path, dbidxs):
        dbidxs = pr.FrozenBitMap(dbidxs)
        assert not os.path.exists(subset_path)
        file_meta = self.file_meta[self.file_meta.index.isin(dbidxs)]
        assert file_meta.shape[0] > 0

        os.makedirs(subset_path)
        meta_info = {'parent':self.path}
        json.dump(meta_info, open(f'{subset_path}/meta.json', 'w'))
        file_meta.to_parquet(f'{subset_path}/file_meta.parquet')
        return SeesawDatasetSubset.load_from_path(self, path=subset_path)

    def create_named_subset(self, subset_name, dbidxs):
        subset_path = f'{self.path}/subsets/{subset_name}'
        return self._create_subset(subset_path, dbidxs)

    def load_subset(self, subset_name) -> BaseDataset:
        return SeesawDatasetSubset.load_from_path(self, f'{self.path}/subsets/{subset_name}')
    
    def get_url(self, dbidx, host='/') -> str:
        path= f'{host}/{self.image_root}/{self.paths[dbidx]}'
        ## remove any extra slashes etc
        return os.path.normpath(path)

    def show_image(self, dbidx, host='/'):
        from IPython.display import Image
        url = self.get_url(dbidx)
        return Image(url=f'{host}/{url}')


    def as_ray_dataset(self, limit=None, parallelism=-1) -> ray.data.Dataset:
        """ with schema {'dbidx', 'file_path, 'bytes'}
            and note: path is in self.paths
        """
        from ray.data.datasource.file_meta_provider import DefaultFileMetadataProvider

        real_prefix = f"{os.path.realpath(self.image_root)}/"
        read_paths = (real_prefix + self.paths).tolist()
        read_paths = read_paths[:limit]
        fix_map = self.dbidx_map
        
        def fix_path(batch_df, fix_map):
            paths = batch_df.path
            fixed_paths = paths.map(lambda x : x[len(real_prefix):])
            
            batch_df = batch_df.assign(file_path=fixed_paths)
            dbidxs = batch_df.file_path.map(lambda p : fix_map[p])
            batch_df = batch_df.assign(dbidx=dbidxs)
            return batch_df[['dbidx', 'file_path', 'bytes']]
            
        binaries = ray.data.read_binary_files(paths=read_paths, include_paths=True, 
                    parallelism=parallelism, meta_provider=DefaultFileMetadataProvider())
    
        return binaries.map_batches(fix_path, batch_format='pandas', fn_kwargs=dict(fix_map=fix_map))

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
        cached_filename = f'{self.dataset_root}/ground_truth/box_data.cached.parquet'
        orig_filename =   f'{self.dataset_root}/ground_truth/box_data.parquet'

        annfolder = f"{self.dataset_root}/ground_truth/annotations"
        if os.path.exists(annfolder):
            annotation_files = get_subdirs_chronological(annfolder)
        else:
            annotation_files = []

        if len(annotation_files) > 0:
            print('found ammended annotations...')
            final_path = cached_filename
            if os.path.exists(cached_filename):
                folder_mtime = os.stat(annfolder).st_mtime
                cache_mtime = os.stat(cached_filename).st_mtime
                compute_combined = cache_mtime < folder_mtime
            else:
                compute_combined = True


            if compute_combined:
                print('computing and caching consolidated annotations')
                box_data = get_parquet(orig_filename, parallelism=0, cache=False)
                for sess_path in annotation_files:
                    box_data = ammend_annotations(image_root=self.image_root, box_data=box_data, annotation_session_path=sess_path)

                ## fix categorical after breaking it before
                box_data = box_data.assign(category=pd.Categorical(box_data.category))
                box_data = fix_dbidx(box_data, self.paths)
                box_data.to_parquet(cached_filename)
                print('done caching.')
            else:
                print('using cached annotations')

        else:
            print(f'no ammended annotations found in {annfolder}. using original version')
            final_path = orig_filename

        box_data = get_parquet(
            final_path,
            parallelism=0, cache=True
        )
        box_data = box_data.assign(origin=final_path)

        if len(annotation_files) > 0:
            return box_data, get_default_qgt(self, box_data)

        qgt_path = f"{self.dataset_root}/ground_truth/qgt.parquet"
        if os.path.exists(qgt_path):
            qgt = get_parquet(qgt_path, parallelism=0, cache=True)
            box_data, qgt = prep_ground_truth(self.paths, box_data, qgt)
            return box_data, qgt
        else: ## note: this will be wrong for subsets. TODO fix this.
            return box_data, get_default_qgt(self, box_data)

    ## TODO: add subset method that takes care of ground truth, and URLS, and that 
    ### can be used by indices over the subset?
from seesaw.util import transactional_folder

def create_dataset(image_src, output_path, paths=[], force=False) -> SeesawDataset:
    """
    if not given explicit paths, it assumes every jpg, jpeg and png is wanted
    """
    with transactional_folder(output_path, force=force) as tmp_output_path:
        image_src = resolve_path(image_src)
        assert os.path.isdir(image_src)

        image_path = f"{tmp_output_path}/images"
        os.symlink(image_src, image_path)
        if len(paths) == 0:
            paths = list_image_paths(image_src)

        df = pd.DataFrame({'dbidx':np.arange(len(paths)), "file_path": paths})
        df.to_parquet(f"{tmp_output_path}/file_meta.parquet")

        _ = SeesawDataset(tmp_output_path) # test read, abort if not well formed

    return SeesawDataset(output_path)
            
### subset. need some way for tools to get path so they can build more stuff underneath
### index subset (easily computed)
### ground truth subset (easily computed)
### pre-built data structures for subset vectors: (eg annoy index, knn graphs, etc. loaded. require some type of name for folder)
### a way to build the first time -> from_dbidxs need some subset paths
### a way to specify where to save intermediates -> 
### a way to re-build later -> need some name to find previously saved stuff.

### goal: it should be easy to know given an index
##  where to save knn indices and where to load them from.

### what we can do in one hour:
## 1. move the data to a folder structure compatible with this idea
##  {dspath}/subsets/{subset_name}/indices/multiscale/knn_graph
##  {dspath}/subsets/{subset_name}/indices/coarse/knn_graph

## 2. change read path to use this rather than gdm to figure out where to read.
#### it makes it easy to handle subsets of other datasets if we want to try that (eg making classes rarer)
import pyroaring as pr
from seesaw.vector_index import VectorIndex

class SeesawDatasetSubset(BaseDataset):
    def __init__(self, parent_dataset, file_meta, path=None):
        """ use factory methods to build
        """ 
        self.path = path
        self.image_root = parent_dataset.image_root
        self.file_meta = file_meta ## this file_meta could be confusing for any code that indexes into file meta using dbidx
        self.paths = self.file_meta['file_path'].values
        self.parent = parent_dataset
        self.dbidxs = pr.FrozenBitMap(self.file_meta.index.values)

    def size(self):
        return self.file_meta.shape[0]
        
    @staticmethod
    def load_from_path(parent_ds, path):
        file_meta = pd.read_parquet(f'{path}/file_meta.parquet')
        os.makedirs(f'{path}/indices/', exist_ok=True)
        return SeesawDatasetSubset(parent_ds, file_meta, path=path)

    def list_indices(self):
        parent_indices = self.parent.list_indices()        
        child_indices = os.listdir(f'{self.path}/indices/')
        return set(parent_indices + child_indices)

    def load_index(self, index_name, *, options):
        if index_name in self.parent.list_indices():
            parent_index = self.parent.load_index(index_name, options=options)
            subset = parent_index.subset(self.dbidxs)
            ## make the path work for other stuff
            subset.path = f'{self.path}/indices/{index_name}/'

            if options.get('use_vec_index', False):
                fullpath = f"{subset.path}/vectors.annoy"
                if os.path.exists(fullpath):
                    print(f'found subset vec index at "{fullpath}"... loading')    
                    vec_index = VectorIndex(load_path=fullpath, prefault=True)
                    subset.vec_index = vec_index        
            return subset
        else:
            raise NotImplementedError

    def load_ground_truth(self):
        boxes, qgt = self.parent.load_ground_truth()
        boxes = boxes[boxes.dbidx.isin(self.dbidxs)]
        qgt = qgt[qgt.index.isin(self.dbidxs)]
        return boxes, qgt

    def load_subset(self, subset_name):
        raise NotImplementedError('not supporting recursive subset right now')

    def get_url(self, dbidx, host='localhost.localdomain:10000') -> str:
        ## url should be same as before
        return self.parent.get_url(dbidx, host)

    def as_ray_dataset(self, limit=None, parallelism=-1) -> ray.data.Dataset:
        raise NotImplementedError()
    

class LeaveOneOutSubset(BaseDataset):
    def __init__(self, parent_dataset : BaseDataset, excluded_dbidx):
        self.parent = parent_dataset
        self.excluded_dbidx = excluded_dbidx
        self.image_root = parent_dataset.image_root
        self.dbidxs = pr.FrozenBitMap(self.parent.file_meta.index.values) - pr.BitMap(excluded_dbidx)

    def size(self):
        return len(self.dbidxs)
        
    def list_indices(self):
        return self.parent.list_indices()        

    def load_index(self, index_name, *, options):
        options['excluded'] = self.excluded_dbidx
        return self.parent.load_index(index_name, options=options)

    def load_ground_truth(self):
        boxes, qgt = self.parent.load_ground_truth()
        boxes = boxes[~boxes.dbidx.isin([self.excluded_dbidx])]
        qgt = qgt[~qgt.index.isin([self.excluded_dbidx])]
        return boxes, qgt

    def load_subset(self, subset_name):
        raise NotImplementedError('not supporting recursive subset right now')

    def get_url(self, dbidx, host='localhost.localdomain:10000') -> str:
        ## url should be same as before
        return self.parent.get_url(dbidx, host)

    def as_ray_dataset(self, limit=None, parallelism=-1) -> ray.data.Dataset:
        raise NotImplementedError()
