import ray
from seesaw import SessionManagerActor, WebSeesaw
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
parser.add_argument('--num_cpus', type=int, default=2, help='cpus assigned to worker')
parser.add_argument('--no_block',  action='store_true', help='start server without blocking')

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
assert os.path.isdir(args.seesaw_root)

ray.init('auto', namespace="seesaw", log_to_driver=True)
serve.start() # started in init_spc.sh

seesaw_root = os.path.abspath(os.path.expanduser(args.seesaw_root))
save_path = os.path.abspath(os.path.expanduser(args.save_path))

session_manager = (SessionManagerActor
                    .options(name='session_manager')
                    .remote(root_dir=seesaw_root, save_path=save_path, num_cpus_per_session=args.num_cpus))

WebSeesaw.deploy(session_manager)
print('new session server deployment is ready, visit it through http://localhost:9000')
if not args.no_block:
  while True:
    input()