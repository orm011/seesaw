from seesaw.dataset_manager import GlobalDataManager
from seesaw.preproc_utils import decode_image_batch, gt_patch_preprocessor, Processor
import seesaw.box_utils
import seesaw.test_box_utils
import json

# from seesaw.box_utils import Segment, BoxBatch, BoundingBoxBatch, BoxOverlay
from ray.data.dataset import ActorPoolStrategy
import transformers

import argparse
import os
import shutil
parser = argparse.ArgumentParser(description="start a seesaw session server")
parser.add_argument("--output_path", type=str, help="folder to save sessions in")
parser.add_argument("--limit", type=int, default=None, help="limit for test run ")
args = parser.parse_args()

if os.path.exists(args.output_path):
    print('path already exists')

tmp_path = args.output_path + '.tmp'
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)

os.makedirs(tmp_path)

import ray
ray.init('auto', namespace='seesaw')

gdm = GlobalDataManager('/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/')
lvisds = gdm.get_dataset('lvis')

config = transformers.CLIPConfig.from_pretrained('/data1/groups/fastai/omoll/seesaw_root2/models/clip-vit-base-patch32/')
mod0 = transformers.CLIPModel.from_pretrained('/data1/groups/fastai/omoll/seesaw_root2/models/clip-vit-base-patch32/')
mod0weights = mod0.state_dict()
weight_ref = ray.put(mod0weights)

# def model_path(index_path):
#     meta = json.load(open(f'{index_path}/info.json', 'r'))
#     return meta['model']

# # lvis_index = gdm.load_index('lvis', 'multiscale')
# mpath = model_path('/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/data/lvis/indices/multiscale/')
gt, qgt = lvisds.load_ground_truth()
gt_ref = ray.put(gt)

remote_args = dict(num_cpus=6, num_gpus=1)
fn_kwargs = dict(config=config, weight_ref=weight_ref, input_col='crop', output_col='vector', num_cpus=remote_args['num_cpus'])

ds = (lvisds.as_ray_dataset(limit=args.limit, parallelism=-1)
        .repartition(100).window(blocks_per_window=10).repartition_each_window(100)
        .map_batches(decode_image_batch(input_col='binary', output_col='image', drop=True), batch_size=50)
        .map_batches(gt_patch_preprocessor(gt_ref, padding=60))
        .map_batches(Processor, compute=ActorPoolStrategy(2, 20, max_tasks_in_flight_per_actor=4), batch_size=200,
                     fn_constructor_kwargs=fn_kwargs, **remote_args)
        .repartition_each_window(num_blocks=1)
        .write_parquet(tmp_path) # 100 parts?

)

os.rename(tmp_path, args.output_path)