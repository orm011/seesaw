import numpy as np
import pandas as pd
from .dataset_manager import GlobalDataManager
from .seesaw_session import LoopParams, SeesawLoop

from typing import Optional, List
from pydantic import BaseModel
import os
import time

def get_image_paths(image_root, path_array, idxs):
    return [ os.path.normpath(f'{image_root}/{path_array[int(i)]}').replace('//', '/') for i in idxs ]

class Box(BaseModel):
    x1 : float
    y1 : float
    x2 : float
    y2 : float
    category : Optional[str] # used for sending ground truth data only, so we can filter by category on the client.

class Imdata(BaseModel):
    url : str
    dbidx : int
    boxes : Optional[List[Box]] # None means not labelled (neutral). [] means positively no boxes.
    refboxes : Optional[List[Box]]


class SessionState:
    current_dataset : str
    current_index : str
    loop : SeesawLoop
    acc_indices : list
    ldata_db : dict
    timing : list

    def __init__(self, gdm : GlobalDataManager, dataset_name, index_name):
        self.gdm = gdm
        self.dataset = self.gdm.get_dataset(dataset_name)
        self.current_dataset = self.dataset.dataset_name
        self.current_index = index_name
        self.acc_indices = []
        self.ldata_db = {}
        self.init_q = None
        self.timing = []

        self.params = LoopParams(interactive='pytorch', warm_start='warm', batch_size=3, 
                minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
                tqdm_disabled=True, granularity='multi', positive_vector_type='vec_only', 
                 num_epochs=2, n_augment=None, min_box_size=10, model_type='multirank2', 
                 solver_opts={'C': 0.1, 'max_examples': 225, 'loss_margin': 0.05})

        self.hdb = gdm.load_index(dataset_name, index_name)
        self.loop = SeesawLoop(self.hdb, params=self.params)

    def step(self):
        start = time.time()
        idxbatch = self.loop.next_batch()

        delta = time.time() - start

        self.acc_indices.append(idxbatch)
        self.timing.append(delta)

        return idxbatch

    def text(self, key):        
        self.init_q = key
        self.loop.set_vec(qstr=key)
        return self.step()

    def get_state(self):
        gdata = []
        for indices in self.acc_indices:
            imdata = self.get_panel_data(idxbatch=indices)
            gdata.append(imdata)
        
        dat = {'gdata':gdata}
        dat['current_dataset'] = self.current_dataset
        dat['current_index'] = self.current_index
        dat['timing']  = self.timing
        # if self.ev.query_ground_truth is not None:
        #     dat['reference_categories'] = self.ev.query_ground_truth.columns.values.tolist()
        # else:
        dat['reference_categories'] = []
        return dat

    def get_panel_data(self, *, idxbatch):
        reslabs = []
        urls = get_image_paths(self.dataset.image_root, self.dataset.paths, idxbatch)

        for (url, dbidx) in zip(urls, idxbatch):
            dbidx = int(dbidx)

            if False: #and (self.ev.box_data is not None):
                bx = self.ev.box_data
                rows = bx[bx.dbidx == dbidx]
                rows = rows[['x1', 'x2', 'y1', 'y2', 'category']]
                refboxes = rows.to_dict(orient='records')
            else:
                refboxes = []

            boxes = self.ldata_db.get(dbidx,None) # None means no annotations yet (undef), empty means no boxes.
            elt = Imdata(url=url, dbidx=dbidx, boxes=boxes, refboxes=refboxes)
            reslabs.append(elt)
        return reslabs

    def update_labeldb(self, gdata):
        for ldata in gdata:
            for imdata in ldata:
                if imdata.boxes is not None:
                    self.ldata_db[imdata.dbidx] = [b.dict() for b in imdata.boxes]
                else:
                    if imdata.dbidx in self.ldata_db:
                        del self.ldata_db[imdata.dbidx]

    def summary(self):
        return {}