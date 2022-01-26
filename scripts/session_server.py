import ray
from fastapi import FastAPI
from seesaw import add_routes
import os
import argparse
from ray import serve

parser = argparse.ArgumentParser(description='start a seesaw session server')
parser.add_argument('--seesaw_root', type=str, help='Seesaw root folder')
parser.add_argument('--save_path', type=str, help='folder to save sessions in')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
assert os.path.isdir(args.seesaw_root)

ray.init('auto', namespace="seesaw")
serve.start(http_options={'port':8000})

app = FastAPI()
WebSeesaw = add_routes(app)

deploy_options = dict(name="seesaw_deployment", ray_actor_options={"num_cpus": 8}, route_prefix='/')
WebSeesawServe = serve.deployment(**deploy_options)(serve.ingress(app)(WebSeesaw))
WebSeesawServe.deploy(root_dir=args.seesaw_root, save_path=args.save_path)

print('sessionserver is ready. visit it through http://localhost:9000')
while True: # wait 
    input()