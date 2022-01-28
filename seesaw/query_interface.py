import numpy as np
import pyroaring as pr
import importlib

def get_constructor(cons_name : str):
    pieces = cons_name.split('.', maxsplit=-1)
    index_mod = importlib.import_module('.'.join(pieces[:-1]))
    constructor = getattr(index_mod, pieces[-1])
    return constructor

class AccessMethod:
    def string2vec(self, string : str) -> np.ndarray:
        raise NotImplementedError('implement me')

    def query(self, *, vector : np.ndarray, topk : int, exclude : pr.BitMap = None) -> np.ndarray:
        raise NotImplementedError('implement me')

    def new_query(self):
        raise NotImplementedError('implement me')

    def subset(self, indices : pr.BitMap):
        raise NotImplementedError('implement me')

    @staticmethod
    def from_path(gdm, index_subpath : str, model_name :str):
        raise NotImplementedError('implement me')

    @staticmethod
    def restore(gdm, type_name : str,  data_path : str, model_name : str):
        c = get_constructor(type_name)
        return c.from_path(gdm, data_path, model_name)

from pydantic import BaseModel
from typing import Optional, List
import pandas as pd

class Box(BaseModel):
    x1 : float
    y1 : float
    x2 : float
    y2 : float
    
class LabelDB:
  def __init__(self):
    self.ldata = {}

  @property
  def seen(self):
    return pr.BitMap(self.ldata.keys()) 

  def put(self, dbidx : int, boxes : List[Box]):
    self.ldata[dbidx] = boxes

  def get(self, dbidx : int, format : str):
    dbidx = int(dbidx)
    if dbidx not in self.ldata:
      return None # has not been seen. used when sending data

    boxes = self.ldata[dbidx]
    if boxes is None: # seen by user but not labeled. consider negative for now
      boxes = []    

    if format == 'df':
      if boxes == []:
        return pd.DataFrame(boxes, columns=['x1', 'x2', 'y1', 'y2']).astype('float32') # cols in case it is empty
      else:
        return pd.DataFrame([b.dict() for b in boxes])[['x1', 'x2', 'y1', 'y2']].astype('float32')

    elif format == 'box':
      return boxes  

class InteractiveQuery(object):
    """
        tracks what has been shown already and supplies it 
        to stateless db as part of query
    """
    def __init__(self, index: AccessMethod):
        self.index = index
        self.returned = pr.BitMap() # images returned from index (not necessarily seen yet)
        self.label_db = LabelDB()
        self.startk = 0

    def query_stateful(self, *args, **kwargs):
        '''
        :param kwargs: forwards arguments to db query method but
         also keeping track of already seen results. also
         keeps track of query history.
        :return:
        '''
        batch_size = kwargs.get('batch_size')
        del kwargs['batch_size']
            
        res =  self.index.query(*args, topk=batch_size, **kwargs, exclude=self.returned, startk=self.startk)
        # assert nextstartk >= self.startk nor really true: if vector changes a lot, 
        self.startk = res['nextstartk']
        self.returned.update(res['dbidxs'])
        del res['nextstartk']
        return res

    def getXy(self):
        raise NotImplementedError('abstract')
