import ray
from seesaw.seesaw_bench import *
from seesaw.textual_feedback_box import std_textual_config

import random
import string
import sys
import math
import argparse



parser = argparse.ArgumentParser('runs benchmark tests and stores metrics in folder')
parser.add_argument('--debug', action='store_true', help='run the script locally to attach a debugger')
parser.add_argument('--limit', type=int, default=None, help='limit the number of benchmarks run (per dataset), helpful for debug')
args = parser.parse_args()

ray.init('auto', namespace='seesaw', log_to_driver=False, ignore_reinit_error=True)


gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/')
os.chdir(gdm.root)

s0 = dict(warm_start='warm', model_type='cosine',
                  batch_size=3, minibatch_size=10,learning_rate=.005,
                  num_epochs=2,loss_margin=.1,max_examples=500)
b0 = dict(n_batches=200,max_feedback=None,box_drop_prob=0., max_results=5)

variants = [
    # dict(name='seesaw', interactive='pytorch', index_name='multiscale'),
    # dict(name='multi', interactive='plain', index_name='multiscale'),
    # dict(name='baseline', interactive='plain', index_name='coarse'),
    # dict(name='refine', interactive='pytorch', index_name='coarse'),
    dict(name='textual_multi', interactive='textual', index_name='multiscale', method_config=std_textual_config),
]

# datasets = ['data/lvis/', 'data/bdd/', 'data/coco/', 'data/dota/', 'data/objectnet/']
datasets = ['data/lvis/', 'data/objectnet/']

nclasses = math.inf if args.limit is None else args.limit
cfgs = gen_configs(gdm, datasets=datasets, variants=variants, s_template=s0, b_template=b0, max_classes_per_dataset=nclasses)
random.shuffle(cfgs)

print(f'{len(cfgs)} generated')

key = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
results_dir = f'/home/gridsan/omoll/bench_results/bench_{key}/'
os.makedirs(results_dir, exist_ok=True)
print(f'outputting benchmark results to file:/{results_dir}')


if args.debug:
  br = BenchRunner(gdm.root, results_dir=results_dir)
  for cfg in cfgs:
    br.run_loop(*cfg)
  sys.exit()

actors = make_bench_actors(resources_per_bench=dict(num_cpus=16, memory=12*(2**30)), 
                          bench_constructor_args=dict(seesaw_root=gdm.root, results_dir=results_dir))
_ = ray.get([a.ready.remote()  for a in actors])
parallel_run(actors=actors, tups=cfgs)