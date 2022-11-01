import argparse
import os
import ray
ray.init('auto', namespace='seesaw')
from seesaw.dataset_manager import GlobalDataManager
from seesaw.knn_graph import compute_knn_from_nndescent
from seesaw.util import reset_num_cpus


root = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/'
gdm = GlobalDataManager(root)
# lvisds = gdm.get_dataset('lvis')
# index = lvisds.load_index('coarse', options=dict())
# _, qgt = lvisds.load_ground_truth()
import pyroaring as pr

knng_name = 'nndescent60'
n_neighbors = 60

def build_and_save_knng(idx, *, knng_name, n_neighbors, num_cpus, low_memory):
    final_path = idx.get_knng_path(knng_name)
    df = compute_knn_from_nndescent(idx.vectors, n_neighbors=n_neighbors, n_jobs=num_cpus, low_memory=low_memory)
    os.makedirs(final_path, exist_ok=True)
    df.to_parquet(f'{final_path}/forward.parquet')
    print('done saving to ', final_path)


class KNNMaker:
    def __init__(self, *, num_cpus):
        reset_num_cpus(num_cpus)
        self.num_cpus = num_cpus
        gdm = GlobalDataManager(root)
        self.gdm = gdm
        ## brings index out but not used

    def __call__(self, batch):
        for (dataset_name, index_name, subset_name) in batch:
            print(f'{dataset_name=} {index_name=} {subset_name=}')
            ds = gdm.get_dataset(dataset_name)
            if subset_name is not None:
                ds = ds.load_subset(subset_name)
            index = ds.load_index(index_name, options=dict(use_vec_index=False))
            build_and_save_knng(index, knng_name=knng_name, n_neighbors=n_neighbors, 
                    num_cpus=self.num_cpus, low_memory=False)
        return batch


combinations = []
for dataset_name in ('lvis',):
    for index_name in ('multiscalemed',):
        combinations.append((dataset_name, index_name, None))

# all_subsets = qgt.columns.values
ds = ray.data.from_items(combinations, parallelism=min(len(combinations), 1))

num_cpus = 90
fn_args = dict(num_cpus=num_cpus)
ray_remote_args=dict(num_cpus=num_cpus, num_gpus=0)
from ray.data.dataset import ActorPoolStrategy

x = ds.map_batches(KNNMaker, batch_size=1, compute=ActorPoolStrategy(1, 1), fn_constructor_kwargs=fn_args, **ray_remote_args).take_all()
