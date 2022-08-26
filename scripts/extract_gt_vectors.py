from seesaw.dataset_manager import GlobalDataManager
import seesaw.preproc_utils
from seesaw.preproc_utils import decode_image_batch, gt_patch_preprocessor, Processor
import seesaw.box_utils
import seesaw.test_box_utils
import json

# from seesaw.box_utils import Segment, BoxBatch, BoundingBoxBatch, BoxOverlay
from ray.data.dataset import ActorPoolStrategy

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

def model_path(index_path):
    meta = json.load(open(f'{index_path}/info.json', 'r'))
    return meta['model']

# lvis_index = gdm.load_index('lvis', 'multiscale')
mpath = model_path('/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/data/lvis/indices/multiscale/')
gt, qgt = lvisds.load_ground_truth()

gt.index = gt.index.rename('box_id')
gt = gt.reset_index()

gt2 = gt[['box_id', 'category', 'x1', 'y1', 'x2', 'y2', 'im_height', 'im_width', 'dbidx']]
gt2ref = ray.put(gt2)

remote_args = dict(num_cpus=4, num_gpus=1)
fn_kwargs = dict(model_path=mpath, input_col='crop', output_col='vector', num_cpus=remote_args['num_cpus'])

ds = (lvisds.as_ray_dataset(limit=args.limit, parallelism=-1)
        .repartition(400)
        .window(blocks_per_window=40)
        .map_batches(decode_image_batch(input_col='binary', output_col='image', drop=True), batch_size=50)
        .map_batches(gt_patch_preprocessor(gt2ref), batch_size=100)
        .map_batches(Processor, compute=ActorPoolStrategy(2, 20, max_tasks_in_flight_per_actor=4), batch_size=200,
                     fn_constructor_kwargs=fn_kwargs, **remote_args)
        .repartition_each_window(num_blocks=10)
        .write_parquet(tmp_path)
)

os.rename(tmp_path, args.output_path)