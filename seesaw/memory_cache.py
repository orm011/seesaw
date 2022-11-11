import ray
import time
import ray
import ray.actor
from typing import Union, Dict

class WrappedRef:
    def __init__(self, ref):
        assert isinstance(ref, ray.ObjectRef)
        self.ref = ref


class ReferenceCache:
    mapping: Dict[str, Union[int, WrappedRef]]
    """
    global store for reference to paths (outlives any single session)
    """

    def __init__(self):
        self.mapping = {}

    def ready(self):
        return True

    def get_or_lock(self, key:str) -> Union[int, WrappedRef]:
        ## returns value or locks key
        ans = self.mapping.get(key, -1)
        if ans == -1: # increment counter so the first caller gets a different value
            self.mapping[key] = 0
        return ans

    def put(self, key:str, value: WrappedRef):
        if self.mapping.get(key, -1) != 0:
            print(f'Warning: ignoring illegal put on {key=}')
        
        if value is None or value.ref is None or not isinstance(value.ref, ray.ObjectRef):
            print(f'Warning: value {value=} is not valid')

        self.mapping[key] = value

    def release_if_locked(self, key: str):
        if self.mapping.get(key,-1) == 0:
            print(f"deleting locked entry for {key=}")
            del self.mapping[key]

    def print_map(self):
        print(self.mapping)


class ReferenceCacheStub:
    ## stub methods: identical semantics to remote
    handle: ray.actor.ActorHandle

    def __init__(self, handle):
        self.handle = handle

    def get_or_lock(self, key:str) -> Union[int, WrappedRef]:
        return ray.get(self.handle.get_or_lock.remote(key))

    def put(self, key : str, value : WrappedRef):
        ray.get(self.handle.put.remote(key, value))

    def release_if_locked(self, key : str):
        ray.get(self.handle.release_if_locked.remote(key))


class LocalCache:
    def __init__(self, cache_actor_name):
        handle = ray.get_actor(cache_actor_name)
        self.remote_cache = ReferenceCacheStub(handle)
        self.mapping = {}

    def get_or_initialize(self, key:str, initializer_function):
        while True:
            if key in self.mapping:
                return self.mapping.get(key)
            
            try:
                ans = self.remote_cache.get_or_lock(key)
                if isinstance(ans, WrappedRef): # got remote value
                    obj = ray.get(ans.ref)
                    self.mapping[key] = obj
                elif ans == -1: # got lock
                    print(f'initializing {key=}')
                    obj = initializer_function()
                    obj_ref = ray.put(obj, _owner=self.remote_cache.handle)
                    self.remote_cache.put(key, WrappedRef(obj_ref))
                elif ans == 0: # someone else is initializing
                    print(f'waiting for other process to initialize {key=}')
                    time.sleep(1)
                else:
                    assert False, 'unkown case'
            finally: # in case something went wrong
                self.remote_cache.release_if_locked(key)