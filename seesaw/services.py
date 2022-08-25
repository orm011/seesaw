from .definitions import resolve_path
from .memory_cache import CacheStub
from .models.embeddings import ModelStub, HGWrapper
import pandas as pd
import ray


def get_cache() -> CacheStub:
    return CacheStub("actor#cache")


g_cache = get_cache()


def get_parquet(parquet_path: str, columns = None, ignore_metadata=True) -> pd.DataFrame:
    return g_cache.read_parquet(parquet_path, columns, ignore_metadata=ignore_metadata)


def get_model_actor(model_path: str) -> ModelStub:
    import ray

    model_path = resolve_path(model_path)
    actor_name = f"/model_actor#{model_path}"  # the slash is important
    try:
        ref = ray.get_actor(actor_name)
        return ModelStub(ref)
    except ValueError as e:
        pass  # will create instead

    def _init_model_actor():
        full_path = model_path

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
                name=actor_name,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                lifetime="detached",
            )
            .remote(path=full_path, device=device)
        )

        # wait for it to be ready
        ray.get(r.ready.remote())
        return r

    # we're using the cache just as a lock
    global_cache = get_cache()
    global_cache._with_lock(actor_name, _init_model_actor)

    # must succeed now...
    ref = ray.get_actor(actor_name)
    return ModelStub(ref)
