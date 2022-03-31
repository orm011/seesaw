import ray
from fastapi import FastAPI
from seesaw import add_routes
import os
import argparse
from ray import serve

"""
deploys session server and exits. if it has been run before, when re-run it will re-deploy the current version.
"""

parser = argparse.ArgumentParser(description='start a seesaw session server')
parser.add_argument('--seesaw_root', type=str, help='Seesaw root folder')
parser.add_argument('--save_path', type=str, help='folder to save sessions in')
parser.add_argument('--num_cpus', type=int, default=16, help='cpus assigned to worker')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
assert os.path.isdir(args.seesaw_root)

ray.init('auto', namespace="seesaw", log_to_driver=True)
serve.start() # started in init_spc.sh

app = FastAPI()
WebSeesaw = add_routes(app)

if ray.available_resources().get('GPU', 0) > .5:
  num_gpus = .5
else:
  num_gpus = 0

seesaw_root = os.path.abspath(os.path.expanduser(args.seesaw_root))
save_path = os.path.abspath(os.path.expanduser(args.save_path))


deploy_options = dict(name="seesaw_deployment", 
                    num_replicas=1,
                    ray_actor_options={'num_cpus': args.num_cpus, 'num_gpus':num_gpus}, 
                    route_prefix='/')

WebSeesawServe = serve.deployment(**deploy_options)(serve.ingress(app)(WebSeesaw))
WebSeesawServe.deploy(root_dir=seesaw_root, save_path=save_path, num_cpus=args.num_cpus)
print('new session server deployment is ready, visit it through http://localhost:9000')
