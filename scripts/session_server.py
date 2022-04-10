import ray
from seesaw import SessionManager, app, WebSeesaw
import os
import argparse
from ray import serve
import time
"""
deploys session server and exits. if it has been run before, when re-run it will re-deploy the current version.
"""

parser = argparse.ArgumentParser(description='start a seesaw session server')
parser.add_argument('--seesaw_root', type=str, help='Seesaw root folder')
parser.add_argument('--save_path', type=str, help='folder to save sessions in')
parser.add_argument('--num_cpus', type=int, default=16, help='cpus assigned to worker')
parser.add_argument('--no_block',  action='store_true', help='start server without blocking')

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
assert os.path.isdir(args.seesaw_root)

ray.init('auto', namespace="seesaw", log_to_driver=True)
serve.start() # started in init_spc.sh


if ray.available_resources().get('GPU', 0) > .5:
  num_gpus = .5
else:
  num_gpus = 0

seesaw_root = os.path.abspath(os.path.expanduser(args.seesaw_root))
save_path = os.path.abspath(os.path.expanduser(args.save_path))

session_manager = ray.remote(SessionManager).options(name='session_manager').remote(root_dir=seesaw_root, save_path=save_path, num_cpus=args.num_cpus)
# kept alive by blocking

deploy_options = dict(name="seesaw_deployment", 
                    num_replicas=1,
                    ray_actor_options={'num_cpus': args.num_cpus, 'num_gpus':num_gpus}, 
                    route_prefix='/')

WebSeesawServe = serve.deployment(**deploy_options)(serve.ingress(app)(WebSeesaw))
WebSeesawServe.deploy(session_manager)
print('new session server deployment is ready, visit it through http://localhost:9000')
if not args.no_block:
  while True:
    input()