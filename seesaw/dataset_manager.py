from .memory_cache import CacheStub
from .query_interface import AccessMethod
from .definitions import DATA_CACHE_DIR, resolve_path
import os
import numpy as np

from .vloop_dataset_loaders import EvDataset

import pandas as pd

import ray
import shutil

import pandas as pd
import shutil
from .basic_types import IndexSpec

from ray.data.extensions import TensorArray
from .models.embeddings import ModelStub, HGWrapper
from .dataset import SeesawDatasetManager, create_dataset


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
        return [f"data/{name}/" for name in os.listdir(self.data_root)]

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
        output_path = f"{self.data_root}/{dataset_name}"
        return create_dataset(image_src, ouput_path=output_path)

    def _fetch_dataset_path(self, dataset_name):
        if not dataset_name.startswith("data/"):
            return f"data/{dataset_name}"
        else:
            return dataset_name

    def get_dataset(self, dataset_name) -> SeesawDatasetManager:
        d_path = self._fetch_dataset_path(dataset_name)
        dataset_path = f"{self.root}/{d_path}"
        print(dataset_path, d_path)
        return SeesawDatasetManager(dataset_path, cache=self.global_cache)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.root})"
