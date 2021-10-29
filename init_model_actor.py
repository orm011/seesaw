import ray
import sys
import time

from seesaw import CLIPWrapper

if __name__ == '__main__':    
    ray.init('auto', namespace='seesaw')
    num_gpus = 0
    if num_gpus == 0:
        device = 'cpu'
    else:
        device = 'cuda:0'
    
    model_actor = ray.remote(CLIPWrapper).options(name='clip', lifetime='detached', num_gpus=num_gpus, num_cpus=4).remote(device=device)
    ray.get(model_actor.ready.remote())
    print('successfully inited model...')