import argparse
from seesaw.research.knn_methods import KNNGraph
import os

parser = argparse.ArgumentParser(
    description="create and preprocess dataset for use by Seesaw"
)

args = parser.parse_args()

import ray
ray.init('auto', namespace='seesaw')
from seesaw.dataset_manager import GlobalDataManager

root = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/'
gdm = GlobalDataManager(root)
lvisds = gdm.get_dataset()
index = lvisds.load_index('multiscale', use_index=False)
_, qgt = lvisds.load_ground_truth()


import pyroaring as pr
g_output_path = '/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/lvis_knns3/'
os.makedirs(g_output_path, exist_ok=True)
import shutil


from seesaw.util import reset_num_cpus

class KNNMaker:
    def __init__(self, *, num_cpus):
        reset_num_cpus(num_cpus)
        self.num_cpus = num_cpus

        gdm = GlobalDataManager(root)
        self.gdm = gdm
        lvisds = gdm.get_dataset('lvis')
        self.index = lvisds.load_index('multiscale', use_index=False)
        _, qgt = lvisds.load_ground_truth()
        self.qgt = qgt

    def __call__(self, batch):
        qgt = self.qgt
        print(f'{len(batch)=}')
        for subset_name in batch:
            print(subset_name)
            final_path = f'{g_output_path}/{subset_name}'
            if os.path.exists(final_path):
                print('already exists')
                continue

            mask = ~qgt[subset_name].isna()
            subset_idxs = pr.BitMap(qgt.index[mask].values)
            subset = self.index.subset(subset_idxs)
            knng, _ = KNNGraph.from_vectors(subset.vectors, n_neighbors=120, n_jobs=self.num_cpus, low_memory=True)
            output_path = f'{final_path}.tmp'
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

            knng.save(output_path, num_blocks=4) 
            os.rename(output_path, final_path)
            print('done with ', output_path)
        return batch


all_subsets = qgt.columns.values
ds = ray.data.from_items(all_subsets, parallelism=500)

num_cpus = 16
fn_args = dict(num_cpus=num_cpus)
ray_remote_args=dict(num_cpus=num_cpus)
from ray.data.dataset import ActorPoolStrategy

x = ds.map_batches(KNNMaker, batch_size=5, compute=ActorPoolStrategy(1, 100), fn_constructor_kwargs=ray_remote_args, **ray_remote_args).take_all()
print('done')
