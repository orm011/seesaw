import ray
import sys
import time
import torch

from seesaw import CLIPWrapper
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, default='seesaw')
    args = parser.parse_args()

    ray.init('auto', namespace=args.namespace)
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda:0'
    
    model_actor = ray.remote(CLIPWrapper).options(name='clip#actor', lifetime='detached', num_gpus=1, num_cpus=4).remote(device=device)
    ray.get(model_actor.ready.remote())
    print('successfully inited model...')