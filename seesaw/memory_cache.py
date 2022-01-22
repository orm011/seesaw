import os
import ray
import pandas as pd

class MemoryCache:
  mapping : dict
  '''
    global store for reference to paths (outlives any single session)
  '''
  def __init__(self):
    self.mapping = {}

  def ready(self):
    return True

  def savepath(self, path : str, ref : ray.ObjectRef):
    self.mapping[path] = ref

  def haspath(self, path : str)->bool:
    return path in self.mapping

  def getobject(self, path: str) -> ray.ObjectRef:
    return self.mapping[path]

  def releaseref(self, path : str, ref : ray.ObjectRef):
    pass

class CacheStub:
  handle : ray.actor.ActorHandle
  def __init__(self, actor_name : str):
    self.handle = ray.get_actor(actor_name)

  def savepath(self, path : str, ref : ray.ObjectRef):
    std_path = os.path.normpath(os.path.realpath(path))
    return ray.get(self.handle.savepath.remote(std_path, ref))

  def haspath(self, path : str) -> bool:
    std_path = os.path.normpath(os.path.realpath(path))
    return ray.get(self.handle.haspath.remote(std_path))

  def getobject(self, path: str): # for some reason this returns the actual object... and not a ref like i thought
    std_path = os.path.normpath(os.path.realpath(path))
    return ray.get(self.handle.getobject.remote(std_path))

  def read_parquet(self, path: str):
      if not self.haspath(path):
        if True:#os.path.isdir(path):
          ds = ray.data.read_parquet(path)
          df = pd.concat(ray.get(ds.to_pandas_refs()))
        # else:
        #   df = pd.read_parquet(path)

        print(f'saving reference to {path}')
        ref = ray.put(df)
        self.savepath(path, ref)
      else:
        df = self.getobject(path)
      
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