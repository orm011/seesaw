import ray
import sys
import time
import torch

from seesaw import  HGWrapper
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
    
    model_actors = [ 
                    # ray.remote(CLIPWrapper).options(name='clip#actor', lifetime='detached', 
                    #                 num_gpus=.5, num_cpus=2).remote(device=device),
                    ray.remote(HGWrapper).options(name='clip#actor', lifetime='detached', 
                                    num_gpus=.5, num_cpus=2).remote(path="/home/gridsan/omoll/seesaw_root/models/clip-vit-base-patch32/", device=device),
                    ray.remote(HGWrapper).options(name='birdclip#actor', lifetime='detached', 
                                    num_gpus=.5, num_cpus=2).remote(path='/home/gridsan/omoll/seesaw_root/models/finetuned_birds/',device=device) 
                    ]

    ready = [ma.ready.remote() for ma in model_actors]
    ray.get(ready)
    print('successfully inited models...')