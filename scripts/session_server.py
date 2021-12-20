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
from seesaw import EvDataset, SeesawLoop, LoopParams, ModelService, GlobalDataManager, VectorIndex

import pickle

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

import shutil

class SessionState:
    current_dataset : str
    ev : EvDataset
    loop : SeesawLoop
    acc_indices : list
    ldata_db : dict

    def __init__(self, dataset_name, ev, vectordir):
        self.current_dataset = dataset_name
        self.ev = ev
        self.acc_indices = []
        self.ldata_db = {}
        self.init_q = None
        self.timing = []

        assert vectordir is not None
        if dataset_name == 'lvis':
            dname = 'coco'
        else:
            dname = dataset_name
        vipath = f'{vectordir}/{dname}.annoy'

        if os.path.exists(vipath):
            self.ev.vec_index = VectorIndex(load_path=vipath, prefault=True)
        else:
            print('index path not found, using slow filesystem')
            self.ev.vec_index = VectorIndex(load_path=self.ev.vec_index_path, prefault=True)

        self.params = LoopParams(interactive='pytorch', warm_start='warm', batch_size=3, 
                minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
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
        if self.ev.query_ground_truth is not None:
            dat['reference_categories'] = self.ev.query_ground_truth.columns.values.tolist()
        else:
            dat['reference_categories'] = []

        return dat

    def get_panel_data(self, *, idxbatch):
        reslabs = []
        urls = get_image_paths(self.current_dataset, self.ev, idxbatch)

        for (url, dbidx) in zip(urls, idxbatch):
            dbidx = int(dbidx)

            if self.ev.box_data is not None:
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
    client_data : ClientData

class Res(BaseModel):
    time :float
    client_data : ClientData

class SaveReq(BaseModel):
    client_data : ClientData

import time
import os

class ResetReq(BaseModel):
    dataset: str


@serve.deployment(name="seesaw_deployment", ray_actor_options={"num_cpus": 32}, route_prefix='/')
@serve.ingress(app)
class WebSeesaw:
    def __init__(self, vectordir):
        self.gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/data')
        self.datasets = ['panama_frames3', 'panama_frames_finetune4',]
        # 'bird_guide_224', 'bird_guide_224_finetuned']
        #objectnet', 'dota', 'lvis','coco', 'bdd']
        ## initialize to first one
        self.evs = {}
        print('loading data refs')
        for dsname in self.datasets:
            ev = self._get_ev(dsname)
            self.evs[dsname] = ev
        print('done loading')
        vectordir = vectordir
        assert vectordir is not None
        self.vectordir = vectordir
        self.xclip = ModelService(ray.get_actor('clip#actor'))
        self._reset_dataset(self.datasets[0])

    def _get_ev(self, dataset_name):
        actor = ray.get_actor(f'{dataset_name}#actor')
        ev = ray.get(ray.get(actor.get_ev.remote()))
        return ev

    def _reset_dataset(self, dataset_name):
        ev = self.evs[dataset_name]
        self.state = SessionState(dataset_name, ev, vectordir=self.vectordir)

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
        start = time.time()
        res = self._step(body.client_data)
        delta = time.time() - start
        self.state.timing.append({'oper':'next', 'start':start, 'processing':delta})
        return {'time':delta, 'client_data':res}

    @app.post('/text', response_model=Res)
    def text(self, key : str):
        start = time.time()
        self.state.init_q = key
        self.state.loop.set_vec(qstr=key)
        res = self._step(None)
        delta = time.time() - start
        self.state.timing.append({'oper':'text', 'start':start, 'processing':delta})
        return {'time':delta, 'client_data':res}

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


vectordir = os.environ.get('VECTORDIR', None)
assert vectordir is not None
# pylint: disable=maybe-no-member
WebSeesaw.deploy(vectordir)
while True: # wait.
    input()