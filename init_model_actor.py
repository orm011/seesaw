import ray
import sys
import time
import torch

from seesaw import CLIPWrapper

if __name__ == '__main__':    
    ray.init('auto', namespace='seesaw')
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda:0'
    
    model_actor = ray.remote(CLIPWrapper).options(name='clip', lifetime='detached', num_gpus=1, num_cpus=4).remote(device=device)
    ray.get(model_actor.ready.remote())
    print('successfully inited model...')