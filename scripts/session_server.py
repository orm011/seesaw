import ray

import fastapi
from fastapi import FastAPI

import typing
import pydantic
from typing import Optional, List
from pydantic import BaseModel
from devtools import debug

import numpy as np
import pandas as pd
from seesaw import EvDataset, SeesawLoop, LoopParams, ModelService, GlobalDataManager


import starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

class CustomHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        print('request: ', request)
        print('data: ', request.data)
        response = await call_next(request)
        print('response: ', response)
        return response

# 'middlewares':[Middleware(CustomHeaderMiddleware)]


app = FastAPI()

ray.init('auto', namespace="seesaw")
print('connected to ray.')
from ray import serve

print('starting ray.serve...')
## can be started in detached mode beforehand to enable fast restart
serve.start(http_options={'port':8000}) ##  the documented ways of specifying this are currently failing...
print('started.')

def get_image_paths(dataset_name, ev, idxs):
    return [ f'/data/{dataset_name}/images/{ev.image_dataset.paths[int(i)]}' for i in idxs]

class SessionState:
    current_dataset : str
    ev : EvDataset
    loop : SeesawLoop
    acc_indices : list
    ldata_db : dict

    def __init__(self, dataset_name, ev):
        self.current_dataset = dataset_name
        self.ev = ev
        self.acc_indices = []
        self.ldata_db = {}

        self.params = LoopParams(interactive='pytorch', warm_start='warm', batch_size=3, 
                minibatch_size=10, learning_rate=0.003, max_examples=225, loss_margin=0.1,
                tqdm_disabled=True, granularity='multi', positive_vector_type='vec_only', 
                 num_epochs=2, n_augment=None, min_box_size=10, model_type='multirank2', 
                 solver_opts={'C': 0.1, 'max_examples': 225, 'loss_margin': 0.05})

        self.loop = SeesawLoop(ev, params=self.params)

    def step(self):
        idxbatch = self.loop.next_batch()
        self.acc_indices.append(idxbatch)
        return idxbatch
        
    def get_state(self):
        gdata = []
        for indices in self.acc_indices:
            imdata = self.get_panel_data(idxbatch=indices)
            gdata.append(imdata)
        
        dat = {'gdata':gdata}
        dat['current_dataset'] = self.current_dataset
        dat['reference_categories'] = self.ev.query_ground_truth.columns.values.tolist()
        return dat

    def get_panel_data(self, *, idxbatch):
        reslabs = []
        bx = self.ev.box_data
        urls = get_image_paths(self.current_dataset, self.ev, idxbatch)

        for (url, dbidx) in zip(urls, idxbatch):
            dbidx = int(dbidx)
            rows = bx[bx.dbidx == dbidx]
            rows = rows[['x1', 'x2', 'y1', 'y2', 'category']]
            refboxes = rows.to_dict(orient='records')

            boxes = self.ldata_db.get(dbidx,None) # None means no annotations yet (undef), empty means no boxes.
            if boxes is not None and len(boxes) > 0: # serialize boxes to format
                rows = pd.DataFrame.from_records(boxes)
                rows = rows[['x1', 'x2', 'y1', 'y2']]
                boxes = rows.to_dict(orient='records')

            elt = Imdata(url=url, dbidx=dbidx, boxes=boxes, refboxes=refboxes)
            reslabs.append(elt)
        return reslabs

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

class ClientData(BaseModel): # Using this as a response for every state transition.
    gdata : List[List[Imdata]]
    datasets : List[str]
    current_dataset : str
    reference_categories : List[str]

class NextReq(BaseModel):
    imdata : List[Imdata]

class ResetReq(BaseModel):
    dataset: str

@serve.deployment(name="seesaw_deployment", ray_actor_options={"num_cpus": 32}, route_prefix='/')
@serve.ingress(app)
class WebSeesaw:
    def __init__(self):
        self.gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/data')
        self.datasets = ['objectnet', 'dota', 'lvis','coco', 'bdd']
        ## initialize to first one
        self.evs = {}
        print('loading data refs')
        for dsname in self.datasets:
            ev = self._get_ev(dsname)
            self.evs[dsname] = ev
        print('done loading')
        self.xclip = ModelService(ray.get_actor('clip#actor'))
        self._reset_dataset(self.datasets[0])

    def _get_ev(self, dataset_name):
        actor = ray.get_actor(f'{dataset_name}#actor')
        ev = ray.get(ray.get(actor.get_ev.remote()))
        return ev

    def _reset_dataset(self, dataset_name):
        ev = self.evs[dataset_name]
        self.state = SessionState(dataset_name, ev)

    def _getstate(self):
        s = self.state.get_state()
        s['datasets'] = self.datasets
        return s

    @app.get('/getstate', response_model=ClientData)
    def getstate(self):
        return self._getstate()

    @app.post('/reset', response_model=ClientData)
    def reset(self, r : ResetReq):
        print(f'resetting state with freshly constructed one for {r.dataset}')
        self._reset_dataset(r.dataset)
        return self._getstate()

    def _step(self, body: NextReq):
        if body is not None: ## refinement code
            box_dict = {}
            idxbatch = []
            for elt in body.imdata:
                if elt.boxes is not None:
                    df = pd.DataFrame([b.dict() for b in elt.boxes], 
                                        columns=['x1', 'x2', 'y1', 'y2']).astype('float32') # cols in case it is empty
                    df = df.assign(dbidx=elt.dbidx)
                    box_dict[elt.dbidx] = df
                    idxbatch.append(elt.dbidx)

            self.state.ldata_db.update(box_dict)

            if len(box_dict) > 0:
                print('calling refine...')
                ## should we provide the full box dict?
                self.state.loop.refine(idxbatch=np.array(idxbatch), box_dict=box_dict)
                print('done refining...')
            else:
                print('no new annotations, skipping refinement')

        self.state.step()
        return self._getstate()

    @app.post('/next', response_model=ClientData)
    def next(self, body : NextReq):
        return self._step(body)

    @app.post('/text', response_model=ClientData)
    def text(self, key : str):
        self.state.loop.set_vec(qstr=key) 
        return self._step(body=None)


# pylint: disable=maybe-no-member
WebSeesaw.deploy()
while True: # wait.
    input()