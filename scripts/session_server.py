import ray
from fastapi import FastAPI

from typing import Optional, List
from pydantic import BaseModel

import numpy as np
import pandas as pd

from seesaw import GlobalDataManager
from seesaw.server_session_state import SessionState, Imdata, Box

import pickle
import time
import os

app = FastAPI()

ray.init('auto', namespace="seesaw")
from ray import serve

print('starting ray.serve...')
## can be started in detached mode beforehand to enable fast restart
serve.start(http_options={'port':8000}) ##  the documented ways of specifying this are currently failing...

class ClientData(BaseModel): # Using this as a response for every state transition.
    gdata : List[List[Imdata]]
    datasets : List[str]
    indices : List[str]
    current_dataset : str
    current_index : str
    reference_categories : List[str]
    timing : List[float]

class NextReq(BaseModel):
    client_data : ClientData

class Res(BaseModel):
    client_data : ClientData

class SaveReq(BaseModel):
    client_data : ClientData

class ResetReq(BaseModel):
    dataset: str


@serve.deployment(name="seesaw_deployment", ray_actor_options={"num_cpus": 8}, route_prefix='/')
@serve.ingress(app)
class WebSeesaw:
    def __init__(self):
        self.gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root')
        self.datasets = self.gdm.list_datasets()
        self.indices = None #self.gdm.get_indices(self.datasets[0])
        self._reset_dataset(self.datasets[0])

    def _reset_dataset(self, dataset_name, index_name=None):
        self.indices = self.gdm.list_indices(dataset_name)
        self.state = SessionState(gdm=self.gdm, dataset_name=dataset_name, index_name=self.indices[0])

    def _getstate(self):
        s = self.state.get_state()
        s['datasets'] = self.datasets
        s['indices'] = self.indices
        return s

    @app.get('/getstate', response_model=Res)
    def getstate(self):
        res =  self._getstate()
        return {'client_data':res}

    @app.post('/reset', response_model=Res)
    def reset(self, r : ResetReq):
        print(f'resetting state with freshly constructed one for {r.dataset}')
        self._reset_dataset(r.dataset)
        res = self._getstate()
        return {'client_data':res}

    def _step(self, cdata: ClientData):
        if cdata is not None: ## refinement code
            self.state.update_labeldb(gdata=cdata.gdata)

            box_dict = {}
            idxbatch = []
            for elt in cdata.gdata[-1]:
                boxes = self.state.ldata_db.get(elt.dbidx, None)
                if boxes is not None:
                    df = pd.DataFrame(boxes, columns=['x1', 'x2', 'y1', 'y2']).astype('float32') # cols in case it is empty
                    df = df.assign(dbidx=elt.dbidx)
                    box_dict[elt.dbidx] = df
                    idxbatch.append(elt.dbidx)

            if len(box_dict) > 0:
                print('calling refine...')
                ## should we provide the full box dict?
                self.state.loop.refine(idxbatch=np.array(idxbatch), box_dict=box_dict)
                print('done refining...')
            else:
                print('no new annotations, skipping refinement')

        self.state.step()
        return self._getstate()

    @app.post('/next', response_model=Res)
    def next(self, body : NextReq):
        res = self._step(body.client_data)
        return {'client_data':res}

    @app.post('/text', response_model=Res)
    def text(self, key : str):
        _ = self.state.text(key=key)
        res = self._getstate()
        return {'client_data':res}

    @app.post('/save', response_model=ClientData)
    def save(self, body : SaveReq):
        print('save req')
        cdata = body.client_data
        self.state.update_labeldb(gdata=cdata.gdata)

        summary = {'qstr':self.state.init_q, 
        'ldata_db':self.state.ldata_db,
        'dataset':self.state.current_dataset,
        'indices':self.state.acc_indices,
        'timing':self.state.timing
        }

        base = os.path.realpath(os.curdir)

        print('saving ', summary)
        fname = f'{base}/session_summary_{time.strftime("%Y%m%d-%H%M%S")}.pkl'
        pickle.dump(summary, open(fname, 'wb'))
        print(f'saved in {fname}')
        return self._getstate()
        ### update all labels
        ### save state with time id


WebSeesaw.deploy()
print('sessionserver is ready. visit it through http://localhost:9000')
while True: # wait.
    input()