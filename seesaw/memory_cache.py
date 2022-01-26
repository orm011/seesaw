import os
import ray
import pandas as pd


class WrappedRef:
  def __init__(self, ref):
    assert isinstance(ref, ray.ObjectRef)
    self.ref = ref

class MemoryCache:
  mapping : dict
  '''
    global store for reference to paths (outlives any single session)
  '''
  def __init__(self):
    self.mapping = {}

  def ready(self):
    return True

  def savepath(self, path : str, wrapped_ref : WrappedRef):
    assert self.mapping[path] is None
    assert wrapped_ref is not None
    assert wrapped_ref.ref is not None
    assert isinstance(wrapped_ref.ref, ray.ObjectRef)
    self.mapping[path] = wrapped_ref

  def pathstate(self, path : str, lock=False)->bool:
    if path not in self.mapping:
      if lock:
        self.mapping[path] = None
      return -1 # not loaded. 
    elif self.mapping[path] == None:
      return 0 # loading
    else:
      return 1 # loaded

  def getobject(self, path: str) -> WrappedRef:
    assert path in self.mapping and self.mapping[path] is not None, path
    return self.mapping[path]

  def releaseref(self, path : str, ref : ray.ObjectRef):
    pass

import time

class CacheStub:
  handle : ray.actor.ActorHandle
  def __init__(self, actor_name : str):
    self.local = {}
    self.handle = ray.get_actor(actor_name)

  def savepath(self, path : str, obj):
    print(f'saving reference to {path}')
    std_path = os.path.normpath(os.path.realpath(path))
    assert std_path not in self.local
    ref = ray.put(obj, _owner=self.handle)
    return ray.get(self.handle.savepath.remote(std_path, WrappedRef(ref)))

  def pathstate(self, path: str, lock=False) -> int:
    std_path = os.path.normpath(os.path.realpath(path))
    if std_path in self.local:
      return 1
    
    return ray.get(self.handle.pathstate.remote(std_path, lock))

  def getobject(self, path: str): # for some reason this returns the actual object... and not a ref like i thought
    std_path = os.path.normpath(os.path.realpath(path))
    if std_path in self.local:
      return self.local[std_path]
    
    wref =  ray.get(self.handle.getobject.remote(std_path))
    assert isinstance(wref, WrappedRef)
    assert wref is not None
    obj = ray.get(wref.ref)
    self.local[std_path] = obj
    return obj

  def read_parquet(self, path: str):
    while True:
      state = self.pathstate(path, lock=True)
      if state == -1: # only one process will see this
        ds = ray.data.read_parquet(path)
        df = pd.concat(ray.get(ds.to_pandas_refs()))
        self.savepath(path, df)
        assert self.pathstate(path) == 1  
      elif state == 0: # someone else is loading, cannot call yet
        time.sleep(1)
        print(f'waiting on loader for {path}...')
      else: # common case
        df = self.getobject(path)
        break

    return df
      

if __name__ == '__main__':
    actor_name = 'actor#cache'
    ray.init('auto', namespace='seesaw')
    try:
      ray.get_actor(actor_name)
      print('cache actor already exists')
    except ValueError:
      print('creating cache actor')

    h = ray.remote(MemoryCache).options(name=actor_name, num_cpus=1, lifetime='detached').remote()
    r = h.ready.remote()
    ray.get(r)
    print('cache actor ready')