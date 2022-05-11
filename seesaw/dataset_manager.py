from seesaw.memory_cache import CacheStub
from seesaw.query_interface import AccessMethod
from seesaw.definitions import DATA_CACHE_DIR, parallel_copy
import os
import numpy as np
from operator import itemgetter
from .preprocess.preprocessor import Preprocessor

from .vloop_dataset_loaders import EvDataset

import glob
import pandas as pd

import ray
import shutil

import pandas as pd
from pyarrow import parquet as pq
import shutil
import io
from .basic_types import IndexSpec

from ray.data.extensions import TensorArray
from .dataset_tools import ExplicitPathDataset
from .models.embeddings import ModelStub, HGWrapper


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


import json


class SeesawDatasetManager:
    def __init__(self, root, dataset_name, dataset_path, cache):
        """Assumes layout created by create_dataset"""
        self.dataset_name = dataset_name
        self.dataset_root = f"{root}/{dataset_path}"
        file_meta = cache.read_parquet(f"{self.dataset_root}/file_meta.parquet")
        self.file_meta = file_meta
        self.paths = file_meta["file_path"].values
        self.image_root = os.path.realpath(f"{self.dataset_root}/images/")
        self.cache = cache

    def get_pytorch_dataset(self):
        return ExplicitPathDataset(
            root_dir=self.image_root, relative_path_list=self.paths
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dataset_name})"

    def preprocess2(
        self, model_path, archive_path=None, archive_prefix="", pyramid_factor=0.5
    ):
        dataset = self.get_pytorch_dataset()
        jit_path = model_path
        vector_root = self.vector_path()
        if os.path.exists(vector_root):
            i = 0
            while True:
                i += 1
                backup_name = f"{vector_root}.bak.{i:03d}"
                if os.path.exists(backup_name):
                    continue
                else:
                    os.rename(vector_root, backup_name)
                    break

        os.makedirs(vector_root, exist_ok=False)
        sds = self

        real_prefix = f"{os.path.realpath(sds.image_root)}/"
        read_paths = ((real_prefix + sds.paths)).tolist()

        read_paths = [os.path.normpath(p) for p in read_paths]

        # paths = [p.replace('//','/') for p in paths]
        meta_dict = dict(zip(read_paths, zip(sds.paths, np.arange(len(sds.paths)))))
        print(list(meta_dict.keys())[0])

        # ngpus = len(self.actors) #
        # actors = self.actors
        actors = []
        try:
            print("starting actors...")
            ngpus = round(ray.available_resources()["GPU"])
            actors = [
                ray.remote(Preprocessor)
                .options(num_cpus=5, num_gpus=1)
                .remote(jit_path=jit_path, output_dir=vector_root, meta_dict=meta_dict)
                for i in range(ngpus)
            ]

            rds = ray.data.read_binary_files(
                paths=read_paths, include_paths=True, parallelism=400
            ).split(ngpus, locality_hints=actors)

            res_iter = []
            for part_id, (actor, shard) in enumerate(zip(actors, rds)):
                of = actor.extract_meta.remote(shard, pyramid_factor, part_id)
                res_iter.append(of)
            ray.get(res_iter)
            return self
        finally:
            print("shutting down actors...")
            for a in actors:
                ray.kill(a)

    def save_vectors(self, vector_data):
        assert (
            np.sort(vector_data.dbidx.unique()) == np.arange(self.paths.shape[0])
        ).all()
        vector_root = self.vector_path()
        os.makedirs(vector_root, exist_ok=False)
        vector_data.to_parquet(f"{vector_root}/manually_saved_vectors.parquet")

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

    def vector_path(self):
        return f"{self.dataset_root}/indices/{self.index_name}/vectors"

    def ground_truth_path(self):
        gt_root = f"{self.dataset_root}/ground_truth/"
        return gt_root

    def load_vec_table(self):
        ds = ray.data.read_parquet(self.vector_path())
        return ds

    def load_eval_categories(self):
        assert os.path.exists(f"{self.dataset_root}/ground_truth")
        return json.load(open(f"{self.dataset_root}/ground_truth/categories.json"))

    def load_ground_truth(self):
        assert os.path.exists(f"{self.dataset_root}/ground_truth")
        qgt = self.cache.read_parquet(f"{self.dataset_root}/ground_truth/qgt.parquet")
        box_data = self.cache.read_parquet(
            f"{self.dataset_root}/ground_truth/box_data.parquet"
        )
        box_data, qgt = prep_ground_truth(self.paths, box_data, qgt)
        return box_data, qgt

    def load_evdataset(
        self,
        *,
        index_subpath,
        force_recompute=False,
        load_coarse=False,
        load_ground_truth=False,
    ) -> EvDataset:

        vec_path = f"{self.dataset_root}/{index_subpath}/vectors"
        cached_meta_path = f"{self.dataset_root}/{index_subpath}/vectors.sorted.cached"
        coarse_meta_path = f"{self.dataset_root}/{index_subpath}/vectors.coarse.cached"
        vec_index_path = f"{self.dataset_root}/{index_subpath}/vectors.annoy"

        if not os.path.exists(cached_meta_path) or force_recompute:
            if os.path.exists(cached_meta_path):
                shutil.rmtree(cached_meta_path)

            print("computing sorted version of metadata...")
            idmap = dict(zip(self.paths, range(len(self.paths))))

            def assign_ids(df):
                return df.assign(dbidx=df.file_path.map(lambda path: idmap[path]))

            ds = ray.data.read_parquet(
                vec_path,
                columns=["file_path", "zoom_level", "x1", "y1", "x2", "y2", "vectors"],
            )
            df = pd.concat(ray.get(ds.to_pandas_refs()), axis=0, ignore_index=True)
            df = assign_ids(df)
            df = df.sort_values(
                ["dbidx", "zoom_level", "x1", "y1", "x2", "y2"]
            ).reset_index(drop=True)
            df = df.assign(order_col=df.index.values)
            max_zoom_out = df.groupby("dbidx").zoom_level.max().rename("max_zoom_level")
            df = pd.merge(df, max_zoom_out, left_on="dbidx", right_index=True)
            splits = split_df(df, n_splits=32)
            ray.data.from_pandas(splits).write_parquet(cached_meta_path)
            print("done....")
        else:
            print("using cached metadata...")

        print("loading fine embedding...")
        assert os.path.exists(cached_meta_path)
        ds = ray.data.read_parquet(
            cached_meta_path,
            columns=[
                "dbidx",
                "zoom_level",
                "max_zoom_level",
                "order_col",
                "x1",
                "y1",
                "x2",
                "y2",
                "vectors",
            ],
        )
        df = pd.concat(ray.get(ds.to_pandas_refs()), ignore_index=True)
        assert df.order_col.is_monotonic_increasing, "sanity check"
        fine_grained_meta = df[
            ["dbidx", "order_col", "zoom_level", "x1", "y1", "x2", "y2"]
        ]
        fine_grained_embedding = df["vectors"].values.to_numpy()

        if load_coarse:
            if not os.path.exists(coarse_meta_path) or force_recompute:
                if os.path.exists(coarse_meta_path):
                    shutil.rmtree(coarse_meta_path)
                print("computing coarse embedding...")
                coarse_emb = infer_coarse_embedding(df)
                assert coarse_emb.dbidx.is_monotonic_increasing
                coarse_emb.to_parquet(coarse_meta_path)
            else:
                print("using cached version...")

            coarse_df = pd.read_parquet(coarse_meta_path)
            assert coarse_df.dbidx.is_monotonic_increasing, "sanity check"
            embedded_dataset = coarse_df["vectors"].values.to_numpy()
            # assert embedded_dataset.shape[0] == self.paths.shape[0], corrupted images are not in embeddding but yes in files
        else:
            embedded_dataset = None

        if load_ground_truth:
            print("loading ground truth...")
            box_data, qgt = self.load_ground_truth()
        else:
            box_data = None
            qgt = None

        # if os.path.exists(self.index_path()):
        #     vec_index = self.index_path() # start actor elsewhere
        # else:
        vec_index = None

        return EvDataset(
            root=self.image_root,
            paths=self.paths,
            embedded_dataset=embedded_dataset,
            query_ground_truth=qgt,
            box_data=box_data,
            embedding=None,  # model used for embedding
            fine_grained_embedding=fine_grained_embedding,
            fine_grained_meta=fine_grained_meta,
            vec_index_path=None,
            vec_index=vec_index,
        )


def split_df(df, n_splits):
    lendf = df.shape[0]
    base_lens = [lendf // n_splits] * n_splits
    for i in range(lendf % n_splits):
        base_lens[i] += 1

    assert sum(base_lens) == lendf
    assert len(base_lens) == n_splits

    indices = np.cumsum([0] + base_lens)

    start_index = indices[:-1]
    end_index = indices[1:]
    cutoffs = zip(start_index, end_index)
    splits = []
    for (a, b) in cutoffs:
        splits.append(df.iloc[a:b])

    tot = sum(map(lambda df: df.shape[0], splits))
    assert df.shape[0] == tot
    return splits


import pickle


def convert_dbidx(ev: EvDataset, ds: SeesawDatasetManager, prepend_ev: str = ""):
    new_path_df = ds.file_meta.assign(dbidx=np.arange(ds.file_meta.shape[0]))
    old_path_df = pd.DataFrame(
        {"file_path": prepend_ev + ev.paths, "dbidx": np.arange(len(ev.paths))}
    )
    ttab = pd.merge(
        new_path_df,
        old_path_df,
        left_on="file_path",
        right_on="file_path",
        suffixes=["_new", "_old"],
        how="outer",
    )
    assert ttab[ttab.dbidx_new.isna()].shape[0] == 0
    tmp = pd.merge(
        ev.box_data,
        ttab[["dbidx_new", "dbidx_old"]],
        left_on="dbidx",
        right_on="dbidx_old",
        how="left",
    )
    tmp = tmp.assign(dbidx=tmp.dbidx_new)
    new_box_data = tmp[[c for c in tmp if c not in ["dbidx_new", "dbidx_old"]]]
    ds.save_ground_truth(new_box_data)


def infer_qgt_from_boxes(box_data, num_files):
    qgt = box_data.groupby(["dbidx", "category"]).size().unstack(level=1).fillna(0)
    qgt = qgt.reindex(np.arange(num_files)).fillna(0)
    return qgt.clip(0, 1)


def infer_coarse_embedding(pdtab):
    # max_zoom_out = pdtab.groupby('file_path').zoom_level.max().rename('max_zoom_level')
    # wmax = pd.merge(pdtab, max_zoom_out, left_on='file_path', right_index=True)
    wmax = pdtab
    lev1 = wmax[wmax.zoom_level == wmax.max_zoom_level]
    ser = lev1.groupby("dbidx").vectors.mean().reset_index()
    res = ser["vectors"].values.to_numpy()
    normres = res / np.maximum(np.linalg.norm(res, axis=1, keepdims=True), 1e-6)
    return ser.assign(vectors=TensorArray(normres))


"""
Expected data layout for a seesaw root with a few datasets:
/workdir/seesaw_data
├── coco  ### not yet preproceseed, just added
│   ├── file_meta.parquet
│   ├── images -> /workdir/datasets/coco
├── coco5 ### preprocessed
│   ├── file_meta.parquet
│   ├── images -> /workdir/datasets/coco
│   └── meta
│       └── vectors
│           ├── part_000.parquet
│           └── part_001.parquet
├── mini_coco
│   ├── file_meta.parquet
│   ├── images -> /workdir/datasets/mini_coco/images
│   └── meta
│       └── vectors
│           ├── part_000.parquet
│           └── part_001.parquet
"""
import sqlite3


def ensure_db(dbpath):
    conn_uri = f"file:{dbpath}?nolock=1"  # lustre makes locking fail, this should run only from manager though.
    conn = sqlite3.connect(conn_uri, uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS models(
                                            m_id INTEGER PRIMARY KEY,
                                            m_created DATETIME default current_timestamp,
                                            m_name TEXT UNIQUE,
                                            m_path TEXT UNIQUE,
                                            m_constructor TEXT,
                                            m_origin_path TEXT 
                                            )"""
        )

        cur.execute(
            """CREATE TABLE IF NOT EXISTS datasets(
                                            d_id INTEGER PRIMARY KEY, 
                                            d_created DATETIME default current_timestamp,
                                            d_name TEXT UNIQUE,
                                            d_path TEXT UNIQUE,
                                            d_origin_path TEXT
                                            )"""
        )

        cur.execute(
            """CREATE TABLE IF NOT EXISTS indices(
                                            i_id INTEGER PRIMARY KEY,
                                            i_created DATETIME default current_timestamp,
                                            i_name TEXT,
                                            i_constructor TEXT, 
                                            i_path TEXT, 
                                            d_id INTEGER,
                                            m_id INTEGER,
                                            FOREIGN KEY(d_id) REFERENCES datasets(d_id)
                                            FOREIGN KEY(m_id) REFERENCES models(_id)
                                            UNIQUE(d_id, i_name)
                                            )"""
        )
        conn.commit()
    finally:
        conn.close()


class GlobalDataManager:
    global_cache: CacheStub

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.exists(root):
            print(f"creating new root folder at {root}")
            os.makedirs(root)

        self.root = root
        self.data_root = f"{root}/data/"
        self.model_root = f"{root}/models/"
        self.index_root = f"{root}/indices/"
        self.global_cache = CacheStub("actor#cache")

        paths = [self.data_root, self.model_root, self.index_root]
        for p in paths:
            os.makedirs(p, exist_ok=True)

        self.dbpath = f"{self.root}/meta.sqlite"
        self.dburi = f"file:{self.dbpath}?nolock=1&mode=ro"
        ensure_db(self.dbpath)

    def _get_connection(self, url_mode="ro"):
        dburi = f"file:{self.dbpath}?nolock=1&mode={url_mode}"
        return sqlite3.connect(dburi, uri=True)

    def _fetch(self, sql, *args, mode="plain", **kwargs):
        try:
            conn = self._get_connection()
            if mode == "dict":
                conn.row_factory = sqlite3.Row
                tups = conn.execute(sql, *args, **kwargs).fetchall()
                return [dict(tup) for tup in tups]
            elif mode == "plain":
                return conn.execute(sql, *args, **kwargs).fetchall()
            elif mode == "df":
                return pd.read_sql_query(sql, conn)
        finally:
            conn.close()

    def _fetch_unique(self, *args, **kwargs):
        tups = self._fetch(*args, **kwargs)
        assert len(tups) == 1
        return tups[0]

    def list_datasets(self):
        tups = self._fetch(
            """
                    select d_name from datasets
                    order by d_id
        """
        )
        return [t[0] for t in tups]

    def list_indices(self):
        df = self._fetch(
            """
            select d_name, i_name, m_name from indices, datasets, models
                    where datasets.d_id == indices.d_id
                    and models.m_id == indices.m_id
                    order by datasets.d_id, i_id
        """,
            mode="df",
        )

        recs = df.to_dict(orient="records")
        return [IndexSpec(**d) for d in recs]

    def get_index_construction_data(self, dataset_name, index_name):
        return self._fetch_unique(
            """select i_constructor, i_path, m_name from indices,models,datasets 
                        where d_name == ? and i_name == ? 
                            and indices.d_id == datasets.d_id
                            and indices.m_id == models.m_id
                        """,
            (dataset_name, index_name),
        )

    def load_index(self, dataset_name, index_name) -> AccessMethod:
        print("loading index")
        cons_name, data_path, model_name = self.get_index_construction_data(
            dataset_name, index_name
        )
        print("got index data")
        return AccessMethod.restore(self, cons_name, data_path, model_name)

    def _get_model_path(self, model_name: str) -> str:
        return self._fetch_unique(
            """select m_path from models where m_name == ?""", (model_name,)
        )[0]

    def get_model_actor(self, model_name: str):
        actor_name = f"/model_actor#{model_name}"  # the slash is important
        try:
            ref = ray.get_actor(actor_name)
            return ModelStub(ref)
        except ValueError as e:
            pass  # will create instead

        def _init_model_actor():
            m_path = self._get_model_path(model_name)
            full_path = f"{self.root}/{m_path}"

            if ray.cluster_resources().get("GPU", 0) == 0:
                device = "cpu"
                num_gpus = 0
                num_cpus = 8
            else:
                device = "cuda:0"
                num_gpus = 0.5
                num_cpus = 4

            r = (
                ray.remote(HGWrapper)
                .options(
                    name=actor_name,
                    num_gpus=num_gpus,
                    num_cpus=num_cpus,
                    lifetime="detached",
                )
                .remote(path=full_path, device=device)
            )

            # wait for it to be ready
            ray.get(r.ready.remote())
            return r

        # we're using the cache just as a lock
        self.global_cache._with_lock(actor_name, _init_model_actor)

        # must succeed now...
        ref = ray.get_actor(actor_name)
        return ModelStub(ref)

    def create_dataset(self, image_src, dataset_name, paths=[]) -> SeesawDatasetManager:
        """
        if not given explicit paths, it assumes every jpg, jpeg and png is wanted
        """
        assert dataset_name is not None
        assert " " not in dataset_name
        assert "/" not in dataset_name
        assert not dataset_name.startswith("_")
        assert (
            dataset_name not in self.list_datasets()
        ), "dataset with same name already exists"
        image_src = os.path.realpath(image_src)
        assert os.path.isdir(image_src)
        dspath = f"{self.data_root}/{dataset_name}"
        assert not os.path.exists(dspath), "name already used"
        os.mkdir(dspath)
        image_path = f"{dspath}/images"
        os.symlink(image_src, image_path)
        if len(paths) == 0:
            paths = list_image_paths(image_src)

        df = pd.DataFrame({"file_path": paths})

        ## use file name order to keep things intuitive
        df = df.sort_values("file_path").reset_index(drop=True)
        df.to_parquet(f"{dspath}/file_meta.parquet")
        return self.get_dataset(dataset_name)

    def _fetch_dataset_path(self, dataset_name):
        d_path = self._fetch_unique(
            """ 
                select d_path from datasets where d_name == ?
        """,
            (dataset_name,),
        )[0]
        return d_path

    def get_dataset(self, dataset_name) -> SeesawDatasetManager:
        all_ds = self.list_datasets()
        assert dataset_name in all_ds, f"{dataset_name} not found in {all_ds}"
        d_path = self._fetch_dataset_path(dataset_name)
        return SeesawDatasetManager(self.root, dataset_name, d_path, self.global_cache)

    def clone(
        self, ds=None, ds_name=None, clone_name: str = None
    ) -> SeesawDatasetManager:
        assert ds is not None or ds_name is not None
        if ds is None:
            ds = self.get_dataset(ds_name)

        if clone_name is None:
            dss = self.list_datasets()
            for i in range(len(dss)):
                new_name = f"{ds.dataset_name}_clone_{i:03d}"
                if new_name not in dss:
                    clone_name = new_name
                    break

        assert clone_name is not None
        shutil.copytree(
            src=ds.dataset_root, dst=f"{self.data_root}/{clone_name}", symlinks=True
        )
        return self.get_dataset(clone_name)

    def clone_subset(
        self, ds=None, ds_name=None, subset_name: str = None, file_names=None
    ) -> SeesawDatasetManager:

        assert ds is not None or ds_name is not None
        if ds is None:
            ds = self.get_dataset(ds_name)

        dataset = ds
        # 1: names -> indices
        image_src = os.path.realpath(dataset.image_root)
        assert subset_name not in self.list_datasets(), "dataset already exists"

        file_set = set(file_names)
        self.create_dataset(
            image_src=image_src, dataset_name=subset_name, paths=file_names
        )
        subds = self.get_dataset(subset_name)

        def vector_subset(tab):
            vt = tab.to_pandas()
            vt = vt[vt.file_path.isin(file_set)]
            return vt

        if os.path.exists(dataset.vector_path()):
            dataset.load_vec_table().map_batches(vector_subset).write_parquet(
                subds.vector_path()
            )

        if os.path.exists(dataset.ground_truth_path()):
            os.symlink(
                os.path.realpath(dataset.ground_truth_path()),
                subds.ground_truth_path().rstrip("/"),
            )

        return self.get_dataset(subset_name)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.root})"


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


import annoy
import random
import os
import sys
import time


def build_annoy_idx(*, vecs, output_path, n_trees):
    start = time.time()
    t = annoy.AnnoyIndex(512, "dot")  # Length of item vector that will be indexed
    for i in range(len(vecs)):
        t.add_item(i, vecs[i])
    print(f"done adding items...{time.time() - start} sec.")
    t.build(n_trees=n_trees)  # 10 trees
    delta = time.time() - start
    print(f"done building...{delta} sec.")
    t.save(output_path)
    return delta


def build_nndescent_idx(vecs, output_path, n_trees):
    import pynndescent

    start = time.time()
    ret = pynndescent.NNDescent(
        vecs.copy(),
        metric="dot",
        n_neighbors=100,
        n_trees=n_trees,
        diversify_prob=0.5,
        pruning_degree_multiplier=2.0,
        low_memory=False,
    )
    print("first phase done...")
    ret.prepare()
    print("prepare done... writing output...", output_path)
    end = time.time()
    difftime = end - start
    pickle.dump(ret, file=open(output_path, "wb"))
    return difftime


import shutil


class VectorIndex:
    def __init__(self, *, base_dir, load_path, copy_to_tmpdir: bool, prefault=False):
        t = annoy.AnnoyIndex(512, "dot")
        self.vec_index = t
        if copy_to_tmpdir:
            print("cacheing first", base_dir, DATA_CACHE_DIR, load_path)
            actual_load_path = parallel_copy(
                base_dir=base_dir, cache_dir=DATA_CACHE_DIR, rel_path=load_path
            )
        else:
            print("loading directly")
            actual_load_path = f"{base_dir}/{load_path}"

        t.load(actual_load_path, prefault=prefault)
        print("done loading")

    def ready(self):
        return True

    def query(self, vector, top_k):
        assert vector.shape == (1, 512) or vector.shape == (512,)
        idxs, scores = self.vec_index.get_nns_by_vector(
            vector.reshape(-1), n=top_k, include_distances=True
        )
        return np.array(idxs), np.array(scores)


RemoteVectorIndex = ray.remote(VectorIndex)


class IndexWrapper:
    def __init__(self, index_actor: RemoteVectorIndex):
        self.index_actor = index_actor

    def query(self, vector, top_k):
        h = self.index_actor.query.remote(vector, top_k)
        return ray.get(h)


def load_ev(*, gdm, dsname, xclip, load_ground_truth=True, load_coarse=True):
    ds = gdm.get_dataset(dsname)
    evref = (
        ray.remote(
            lambda: ds.load_evdataset(
                load_ground_truth=load_ground_truth, load_coarse=load_coarse
            )
        )
        .options(num_cpus=16)
        .remote()
    )

    # mem_needed = os.stat(ds.index_path()).st_size
    # vi = (RemoteVectorIndex.options(num_cpus=8, memory=mem_needed).remote(load_path=ds.index_path(), copy_to_tmpdir=True, prefault=True))
    # readyref = vi.ready.remote()
    ev = ray.get(evref)
    ev.embedding = xclip
    ev.vec_index = ds.index_path()
    return ev
