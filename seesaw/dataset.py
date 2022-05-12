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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dataset_name})"

    def preprocess2(
        self,
        model_path,
        cpu=False,
        archive_path=None,
        archive_prefix="",
        pyramid_factor=0.5,
    ):
        import ray
        from .multiscale.preprocessor import Preprocessor

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
            ngpus = math.ceil(ray.available_resources().get("GPU", 0))
            nactors = ngpus if ngpus > 0 else 1
            actors = [
                ray.remote(Preprocessor)
                .options(num_cpus=5, num_gpus=(1 if ngpus > 0 else 0))
                .remote(jit_path=jit_path, output_dir=vector_root, meta_dict=meta_dict)
                for i in range(nactors)
            ]

            rds = ray.data.read_binary_files(
                paths=read_paths, include_paths=True, parallelism=400
            ).split(nactors, locality_hints=actors)

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
        return f"{self.dataset_root}/indices/multiscale/vectors"

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

    df = pd.DataFrame({"file_path": paths})
    df.to_parquet(f"{dspath}/file_meta.parquet")

    os.rename(dspath, final_dspath)
    return SeesawDatasetManager(final_dspath)
