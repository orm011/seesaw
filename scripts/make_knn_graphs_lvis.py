import argparse
from seesaw.research.knn_methods import KNNGraph
import os
import ray
ray.init('auto', namespace='seesaw')
from seesaw.dataset_manager import GlobalDataManager

root = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/'
gdm = GlobalDataManager(root)
lvisds = gdm.get_dataset('lvis')
index = lvisds.load_index('coarse', options=dict())
_, qgt = lvisds.load_ground_truth()


import pyroaring as pr
#g_output_path = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/lvis_knns3/'
# os.makedirs(g_output_path, exist_ok=True)
# import shutil


from seesaw.util import reset_num_cpus

class KNNMaker:
    def __init__(self, *, num_cpus):
        reset_num_cpus(num_cpus)
        self.num_cpus = num_cpus
        gdm = GlobalDataManager(root)
        self.gdm = gdm
        self.ds = gdm.get_dataset('lvis')
        self.index = self.ds.load_index('coarse', options=dict())
        ## brings index out but not used

    def __call__(self, batch):
        print(f'{len(batch)=}')
        for subset_name in batch:
            print(subset_name)
            sds = self.ds.load_subset(subset_name)
            idx = sds.load_index('coarse', options=dict())
            final_path = idx.get_knng_path()
            knng, _ = KNNGraph.from_vectors(idx.vectors, n_neighbors=30, n_jobs=self.num_cpus, low_memory=True)

            os.makedirs(final_path, exist_ok=True)
            knng.save(final_path, num_blocks=1) 
            print('done with ', final_path)
        return batch


all_subsets = qgt.columns.values
ds = ray.data.from_items(all_subsets, parallelism=250)

num_cpus = 20
fn_args = dict(num_cpus=num_cpus)
ray_remote_args=dict(num_cpus=num_cpus)
from ray.data.dataset import ActorPoolStrategy

x = ds.map_batches(KNNMaker, batch_size=5, compute=ActorPoolStrategy(1, 100), fn_constructor_kwargs=ray_remote_args, **ray_remote_args).take_all()
print('done')
