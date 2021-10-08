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
    
    model_actor = ray.remote(CLIPWrapper).options(name='clip', lifetime='detached', num_gpus=num_gpus, num_cpus=.1).remote(device=device)

    ## this should work
    _ = ray.get_actor('clip')
    time.sleep(10)
    print('clip done')