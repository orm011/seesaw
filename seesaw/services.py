from seesaw.util import parallel_read_parquet
from .definitions import resolve_path
from .memory_cache import LocalCache
from .models.embeddings import ModelStub, HGWrapper
import pandas as pd

_cache = None # initialize lazily so we can use the same functions 
# with or without a cache. 
def _get_cache() -> LocalCache:
    # import ray
    # ray.init('auto', namespace='seesaw', log_to_driver=False, ignore_reinit_error=True)
    global _cache
    if _cache is None:
        _cache = LocalCache("actor#cache")
    
    return _cache

def _cache_closure(closure, *, key: str, use_cache : bool):
    if use_cache:
        cache = _get_cache()
        return cache.get_or_initialize(key, closure)
    else:
        return closure()

def get_parquet(path: str, columns=None, parallelism=-1, cache=True) -> pd.DataFrame:
    ## todo: columns are a buggy argument for cache
    path = resolve_path(path)
    def _init_fun():
        return parallel_read_parquet(path, columns, parallelism = parallelism)
    return _cache_closure(_init_fun, key=path, use_cache=cache)

def read_state_dict(path: str, jit: bool, use_cache = True) -> dict:
    import torch
    path = resolve_path(path)
    def _init_fun():
        if jit:
            return torch.jit.load(path, map_location="cpu").state_dict()
        else:  # the result of a torch load could already be a state dict, right?
            mod = torch.load(path, map_location="cpu")
            if isinstance(mod, torch.nn.Module):
                return mod.state_dict()
            else:  # not sure what else to do here
                return mod
    
    return _cache_closure(_init_fun, key=path, use_cache=use_cache)

def get_model_actor(model_path : str) -> ModelStub:
    import ray
    model_path = resolve_path(model_path)
    key = f"model_actor#{model_path}" 

    def initializer():
        if ray.cluster_resources().get("GPU", 0) == 0:
            device = "cpu"
            num_gpus = 0
            num_cpus = 8
        else:
            device = "cuda:0"
            num_gpus = 0.5
            num_cpus = 4

        r = (
            ray.remote(HGWrapper)
            .options(
                name=key,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                lifetime="detached",
            )
            .remote(path=model_path, device=device, num_cpus=num_cpus)
        )

        # wait for it to be ready
        ray.get(r.ready.remote())
        return r
    
    model_ref = _cache_closure(initializer, key=key, use_cache=True)
    return ModelStub(model_ref)