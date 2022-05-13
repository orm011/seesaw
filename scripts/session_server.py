import ray
from seesaw.web.session_manager import SessionManagerActor
from seesaw.web.seesaw_app import app
import uvicorn
import os
import argparse

"""
deploys session server and exits. if it has been run before, when re-run it will re-deploy the current version.
"""

import argparse

parser = argparse.ArgumentParser(description="start a seesaw session server")
parser.add_argument("--seesaw_root", type=str, help="Seesaw root folder")
parser.add_argument("--save_path", type=str, help="folder to save sessions in")
parser.add_argument("--num_cpus", type=int, default=2, help="cpus assigned to worker")
parser.add_argument(
    "--no_block", action="store_true", help="start server without blocking"
)

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
assert os.path.isdir(args.seesaw_root)

ray.init("auto", namespace="seesaw", log_to_driver=True)

seesaw_root = os.path.abspath(os.path.expanduser(args.seesaw_root))
save_path = os.path.abspath(os.path.expanduser(args.save_path))

actor_name = "session_manager"
try:
    oldh = ray.get_actor(actor_name)
    print("found old session_manager actor, destroying it (old sessions will be lost)")
    ray.kill(oldh)
except:
    pass

session_manager = SessionManagerActor.options(name=actor_name).remote(
    root_dir=seesaw_root, save_path=save_path, num_cpus_per_session=args.num_cpus
)

ray.get(session_manager.ready.remote())

uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
