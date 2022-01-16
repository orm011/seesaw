import ray
from fastapi import FastAPI

from typing import Optional, List, Dict
from pydantic import BaseModel

import numpy as np
import pandas as pd

from seesaw import GlobalDataManager
from seesaw.server_session_state import Session, Imdata, Box, SessionState

import pickle
import time
import os

app = FastAPI()

ray.init('auto', namespace="seesaw")
from ray import serve

print('starting ray.serve...')
## can be started in detached mode beforehand to enable fast restart
serve.start(http_options={'port':8000}) ##  the documented ways of specifying this are currently failing...

class IndexSpec(BaseModel):
    d_name:str 
    i_name:str
    m_name:Optional[str]

class AppState(BaseModel): # Using this as a response for every state transition.
    indices : List[IndexSpec]
    current_index : IndexSpec
    session : SessionState

class SessionReq(BaseModel):
    client_data : AppState

class ResetReq(BaseModel):
    index : IndexSpec

@serve.deployment(name="seesaw_deployment", ray_actor_options={"num_cpus": 8}, route_prefix='/')
@serve.ingress(app)
class WebSeesaw:
    def __init__(self):
        self.gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root')
        self.indices = self.gdm.list_indices()
        self.current_index = self.indices[0]
        self._reset_dataset(self.current_index)

    def _reset_dataset(self, index_spec):
        self.session = Session(gdm=self.gdm, dataset_name=index_spec['d_name'], index_name=index_spec['i_name'])
        self.current_index = index_spec

    def _getstate(self):
        return AppState(indices=self.indices, 
                            current_index=self.current_index, 
                            session=self.session.get_state())

    @app.get('/getstate', response_model=AppState)
    def getstate(self):
        return self._getstate()

    @app.post('/reset', response_model=AppState)
    def reset(self, r : ResetReq):
        print(f'resetting state with freshly constructed one for {r.index}')
        self._reset_dataset(r.index.dict())
        return self._getstate()

    def _step(self, state: SessionState):
        if state is not None: ## refinement code
            self.session.update_state(state)

            box_dict = {}
            idxbatch = []
            for elt in state.gdata[-1]:
                boxes = self.session.ldata_db.get(elt.dbidx, None)
                if boxes is not None:
                    df = pd.DataFrame(boxes, columns=['x1', 'x2', 'y1', 'y2']).astype('float32') # cols in case it is empty
                    df = df.assign(dbidx=elt.dbidx)
                    box_dict[elt.dbidx] = df
                    idxbatch.append(elt.dbidx)

            if len(box_dict) > 0:
                print('calling refine...')
                ## should we provide the full box dict?
                self.session.loop.refine(idxbatch=np.array(idxbatch), box_dict=box_dict)
                print('done refining...')
            else:
                print('no new annotations, skipping refinement')

        self.session.step()
        return self._getstate()

    @app.post('/next', response_model=AppState)
    def next(self, body : SessionReq):
        return self._step(body.client_data.session)

    @app.post('/text', response_model=AppState)
    def text(self, key : str):
        _ = self.session.text(key=key)
        return self._getstate()

    @app.post('/save', response_model=AppState)
    def save(self, body : SessionReq):
        print('save req')
        self.session.update_state(body.client_data.session)

        summary = {'qstr':self.session.init_q, 
        'ldata_db':self.session.ldata_db,
        'dataset':self.session.current_dataset,
        'indices':self.session.acc_indices,
        'timing':self.session.timing
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