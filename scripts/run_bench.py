import ray
from seesaw.seesaw_bench import *
from seesaw.configs import std_textual_config, std_linear_config

import random
import string
import sys
import math
import argparse



parser = argparse.ArgumentParser('runs benchmark tests and stores metrics in folder')
parser.add_argument('--limit', type=int, default=None, help='limit the number of benchmarks run (per dataset), helpful for debug')
parser.add_argument('--num_actors', type=int, default=None, help='amount of actors created. leave blank for auto. 0 for local')
parser.add_argument('--num_cpus', type=int, default=8, help='number of cpus per actor')
parser.add_argument('--timeout', type=int, default=20, help='how long to wait for actor creation')
parser.add_argument('--exp_name', type=str, default=None, help='some mnemonic for experiment')
args = parser.parse_args()

ray.init('auto', namespace='seesaw', log_to_driver=False, ignore_reinit_error=True)


gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/')
os.chdir(gdm.root)

s0 = dict(batch_size=3, method_config={}, shortlist_size=50)
b0 = dict(n_batches=200, max_feedback=None, box_drop_prob=0., max_results=5, provide_textual_feedback=False)

variants = [
    # dict(name='seesaw', interactive='pytorch', index_name='multiscale', agg_method='avg_score', method_config=std_linear_config),
    # dict(name='seesaw_avg_vec', interactive='pytorch', index_name='multiscale', agg_method='avg_vector', method_config=std_linear_config),

    dict(name='multi', interactive='plain', index_name='multiscale', agg_method='avg_score'),
    dict(name='multi_avg_vec', interactive='plain', index_name='multiscale', agg_method='avg_vector'),

    # dict(name='baseline', interactive='plain', index_name='coarse'),
    # dict(name='refine', interactive='pytorch', index_name='coarse', method_config=std_linear_config),
    dict(name='textual', interactive='textual', index_name='multiscale', 
      agg_method='avg_score', method_config={**std_textual_config, **{'mode':'linear'}}, provide_textual_feedback=True),
]

# datasets = ['data/lvis/', 'data/bdd/', 'data/coco/', 'data/dota/', 'data/objectnet/']
datasets = ['data/lvis/', 'data/objectnet/']

nclasses = math.inf if args.limit is None else args.limit
cfgs = gen_configs(gdm, datasets=datasets, variants=variants, s_template=s0, b_template=b0, max_classes_per_dataset=nclasses)
random.shuffle(cfgs)

print(f'{len(cfgs)} generated')

key = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
exp_name = (args.exp_name + '_' if args.exp_name is not None else '') + key
results_dir = f'/home/gridsan/omoll/bench_results/bench_{exp_name}/'
os.makedirs(results_dir, exist_ok=True)
print(f'outputting benchmark results to file:/{results_dir}')


if args.num_actors == 0:
  br = BenchRunner(gdm.root, results_dir=results_dir)
  for cfg in cfgs:
    br.run_loop(*cfg)
else:
  actors = make_bench_actors(actor_options=dict(name=exp_name, num_cpus=args.num_cpus, memory=10*(2**30)), 
                            bench_constructor_args=dict(seesaw_root=gdm.root, results_dir=results_dir, num_cpus=args.num_cpus),
                            num_actors=args.num_actors,
                            timeout=args.timeout,
                            )
  print(f'made {len(actors)} actors')
  parallel_run(actors=actors, tups=cfgs)