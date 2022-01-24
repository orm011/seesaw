import ray
from seesaw.seesaw_bench import *

ray.init('auto', namespace='seesaw', log_to_driver=False, ignore_reinit_error=True)


gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/')
os.chdir(gdm.root)

s0 = dict(warm_start='warm', model_type='cosine',
                  batch_size=3, minibatch_size=10,learning_rate=.005,
                  num_epochs=2,loss_margin=.1,max_examples=500)
b0 = dict(n_batches=10,max_feedback=None,box_drop_prob=0., max_results=100)

variants = [
    dict(name='seesaw', interactive='pytorch', index_name='multiscale'),
    dict(name='multi', interactive='plain', index_name='multiscale'),
    dict(name='baseline', interactive='plain', index_name='coarse'),
]

datasets = ['data/lvis/', 'data/bdd/', 'data/coco/', 'data/dota/', 'data/objectnet/']

cfgs = gen_configs(gdm, datasets=datasets, variants=variants, s_template=s0, b_template=b0, max_classes_per_dataset=2000)


print(f'{len(cfgs)} generated')
import random
import string

key = ''.join([random.choice(string.ascii_letters) for _ in range(10)])
results_dir = f'/home/gridsan/omoll/bench_results/bench_{key}/'
os.makedirs(results_dir, exist_ok=True)
print(f'outputting benchmark results to file:/{results_dir}')

#br = BenchRunner(seesaw_root=gdm.root, results_dir=results_dir)
## this will init the model so there's no race 
br = BenchRunner(gdm.root, results_dir=results_dir)
br.run_loop(*cfgs[0])

actors = make_bench_actors(resources_per_bench=dict(num_cpus=16, memory=12*(2**30)), 
                           bench_constructor_args=dict(seesaw_root=gdm.root, results_dir=results_dir))

_ = ray.get([a.ready.remote()  for a in actors])                           
print('actors ready')

parallel_run(actors=actors, tups=cfgs[1:])