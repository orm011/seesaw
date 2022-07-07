import os
import pandas as pd
import ray


class WrappedRef:
    def __init__(self, ref):
        assert isinstance(ref, ray.ObjectRef)
        self.ref = ref


class ReferenceCache:
    mapping: dict
    """
    global store for reference to paths (outlives any single session)
  """

    def __init__(self):
        self.mapping = {}

    def ready(self):
        return True

    def savepath(self, path: str, wrapped_ref: WrappedRef):
        assert path in self.mapping
        assert self.mapping[path] is None
        assert wrapped_ref is not None
        assert wrapped_ref.ref is not None
        assert isinstance(wrapped_ref.ref, ray.ObjectRef)
        self.mapping[path] = wrapped_ref

    def pathstate(self, path: str, lock=False) -> bool:
        if path not in self.mapping:
            if lock:
                self.mapping[path] = None
            return -1  # not loaded.
        elif self.mapping[path] == None:
            return 0  # loading
        else:
            return 1  # loaded

    def release(self, path: str):
        if self.pathstate(path, lock=False) == 0:
            print("releasing lock for uninitialized")
            # if half-initialized, remove
            del self.mapping[path]
        else:
            print("release for", path)

    def print_map(self):
        print(self.mapping)

    def getobject(self, path: str) -> WrappedRef:
        assert path in self.mapping and self.mapping[path] is not None, path
        return self.mapping[path]

    def releaseref(self, path: str, ref: ray.ObjectRef):
        pass


import torch
import time


class CacheStub:
    handle: ray.actor.ActorHandle

    def __init__(self, actor_name: str):
        self.local = {}
        self.handle = ray.get_actor(actor_name)

    def savepath(self, path: str, obj):
        print(f"saving reference to {path}")
        std_path = os.path.normpath(os.path.realpath(path))
        assert std_path not in self.local
        ref = ray.put(obj, _owner=self.handle)
        return ray.get(self.handle.savepath.remote(std_path, WrappedRef(ref)))

    def _release(self, path: str):
        std_path = os.path.normpath(os.path.realpath(path))
        return ray.get(self.handle.release.remote(std_path))

    def pathstate(self, path: str, lock=False) -> int:
        std_path = os.path.normpath(os.path.realpath(path))
        if std_path in self.local:
            return 1

        return ray.get(self.handle.pathstate.remote(std_path, lock))

    def getobject(
        self, path: str
    ):  # for some reason this returns the actual object... and not a ref like i thought
        std_path = os.path.normpath(os.path.realpath(path))
        if std_path in self.local:
            return self.local[std_path]

        wref = ray.get(self.handle.getobject.remote(std_path))
        assert isinstance(wref, WrappedRef)
        assert wref is not None
        obj = ray.get(wref.ref)
        self.local[std_path] = obj
        return obj

    def _with_lock(self, path: str, init_fun):
        while True:
            state = self.pathstate(path, lock=True)
            if state == -1:  # only one process will see this
                try:
                    obj = init_fun()
                    self.savepath(path, obj)
                finally:  # in case interrupted/killed etc.
                    self._release(path)
            elif state == 0:  # someone else is loading, cannot call yet
                print(f"{path} is locked by someone else... waiting")
                time.sleep(1)
            elif state == 1:  # common case
                obj = self.getobject(path)
                break
            else:
                assert False, "unknown cache state"

        return obj

    def read_parquet(self, path: str, columns = None):
        def _init_fun():
            ds = None
            if columns is not None: 
                ds = ray.data.read_parquet(path, columns=columns)
            else: 
                ds = ray.data.read_parquet(path)
            df = pd.concat(ray.get(ds.to_pandas_refs()))
            df = df.sort_values('dbidx', axis=0)
            print(df.columns)
            return df

        return self._with_lock(path, _init_fun)

    def read_state_dict(self, path: str, jit: bool):
        def _init_fun():
            if jit:
                return torch.jit.load(path, map_location="cpu").state_dict()
            else:  # the result of a torch load could already be a state dict, right?
                mod = torch.load(path, map_location="cpu")
                if isinstance(mod, torch.nn.Module):
                    return mod.state_dict()
                else:  # not sure what else to do here
                    return mod

        return self._with_lock(path, _init_fun)
