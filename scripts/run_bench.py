import ray
from seesaw.seesaw_bench import *
import random
import string
import math
import yaml
import argparse
import ray.data
from ray.data import ActorPoolStrategy

parser = argparse.ArgumentParser("runs benchmark tests and stores metrics in folder")

parser.add_argument(
    "--dryrun", action="store_true", help="run one benchmark locally per dataset"
)
parser.add_argument("--num_cpus", type=int, default=2, help="number of cpus per actor")

parser.add_argument(
    "--timeout", type=int, default=20, help="how long to wait for actor creation"
)
parser.add_argument(
    "--output_dir", type=str, help="dir where experiment results dir will be created"
)

parser.add_argument("--root_dir", type=str, help="seesaw root dir to use for benchmark")

parser.add_argument(
    "configs", help="yaml file(s) with benchmark configurations (see example_config.yaml)", 
        type=str,  nargs='+'
)

args = parser.parse_args()
assert os.path.isdir(args.root_dir)
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

# assert os.path.isfile(args.CONFIG)
yls = []
for path in args.configs:
    yl = yaml.safe_load(open(path, 'r'))
    yls.append(yl)

ray.init("auto", namespace="seesaw", log_to_driver=False, ignore_reinit_error=True)
gdm = GlobalDataManager(args.root_dir)
os.chdir(gdm.root)

names = set([])
all_cfgs = []
for i,yl in enumerate(yls):
    if args.dryrun:
        nclasses = 1
    else:
        nclasses=math.inf

    variants = yl['variants']
    for v in variants:
        if v["name"] in names:
            assert False, f'repeated variant name "{v["name"]}" will make it harder to compare variants afterwards'
        names.add(v["name"])

    datasets = yl['datasets']
    shared_session_params = yl['shared_session_params']
    shared_bench_params = yl['shared_bench_params']

    cfgs = gen_configs(
        gdm,
        datasets=datasets,
        variants=variants,
        s_template=shared_session_params,
        b_template=shared_bench_params,
        max_classes_per_dataset=nclasses,
    )

    print(f"{len(cfgs)} generated from {args.configs[i]}")

    all_cfgs.extend(cfgs)


key = "".join([random.choice(string.ascii_letters) for _ in range(10)])
exp_name = key
results_dir = f"{args.output_dir}/bench_{exp_name}/"
os.makedirs(results_dir, exist_ok=True)
print(f"outputting benchmark results to file:/{results_dir}")

class BatchRunner:
    def __init__(self, redirect_output=True):
        self.br = BenchRunner(gdm.root, results_dir=results_dir, 
                        num_cpus=args.num_cpus, redirect_output=redirect_output)
        
    def __call__(self, cfgs):
        for cfg in cfgs:
            self.br.run_loop(*cfg)

        return cfgs

if args.dryrun:
    br = BatchRunner(redirect_output=False)
    br(all_cfgs)
else:
    random.shuffle(all_cfgs) # randomize task order to kind-of balance work
    ds = ray.data.from_items(all_cfgs, parallelism=800)
    actor_options = dict(num_cpus=args.num_cpus, memory=5 * (2**30))
    ## use a small batch size so that maybe failures affect as few classes as possible?
    _ = ds.map_batches(BatchRunner, batch_size=1, compute=ActorPoolStrategy(10,300), **actor_options).take_all()

print("done with benchmark. computing summary")
get_all_session_summaries(results_dir, force_recompute=True)
print("done with summary.")
