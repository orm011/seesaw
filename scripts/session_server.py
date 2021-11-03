import ray

import fastapi
from fastapi import FastAPI

import typing
import pydantic
from typing import Optional, List
from pydantic import BaseModel

import numpy as np
import pandas as pd
from seesaw import EvDataset, SeesawLoop, LoopParams, ModelService, GlobalDataManager

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
        self.acc_indices = [np.zeros(0, dtype=np.int32)]
        self.ldata_db = {}

        self.params = LoopParams(interactive='pytorch', warm_start='warm', batch_size=10, 
                minibatch_size=10, learning_rate=0.003, max_examples=225, loss_margin=0.1,
                tqdm_disabled=True, granularity='multi', positive_vector_type='vec_only', 
                 num_epochs=2, n_augment=None, min_box_size=10, model_type='cosine', 
                 solver_opts={'C': 0.1, 'max_examples': 225, 'loss_margin': 0.05})

        self.loop = SeesawLoop(ev, params=self.params)

    def step(self):
        idxbatch = self.loop.next_batch()
        self.acc_indices.append(idxbatch)
        return idxbatch
        
    def get_state(self):
        dat = self.get_panel_data(next_idxs=np.concatenate(self.acc_indices))
        dat['current_dataset'] = self.current_dataset
        dat['reference_categories'] = self.ev.query_ground_truth.columns.values.tolist()
        return dat

    def get_latest(self):
        dat = self.get_panel_data(next_idxs=self.acc_indices[-1])
        return dat

    def get_panel_data(self, *, next_idxs):
        reslabs = []
        for (i,dbidx) in enumerate(next_idxs):
            # boxes = copy.deepcopy(label_db.get(dbidx, None))
            bx = self.ev.box_data
            rows = bx[bx.dbidx == dbidx]
            rows = rows.rename(mapper={'x1': 'xmin', 'x2': 'xmax', 'y1': 'ymin', 'y2': 'ymax'}, axis=1)
            rows = rows[['xmin', 'xmax', 'ymin', 'ymax', 'category']]
            recs = rows.to_dict(orient='records')
            reslabs.append({'value': -1 if rows.shape[0] == 0 else 1, 
                            'id': i, 'dbidx': int(dbidx), 'boxes': recs})


        llabs = []
        for (i,dbidx) in enumerate(next_idxs):
            boxes = self.ldata_db.get(dbidx,[])
            recs = []
            if len(boxes) > 0:
                rows = pd.DataFrame.from_records(boxes)
                rows = rows.rename(mapper={'x1': 'xmin', 'x2': 'xmax', 'y1': 'ymin', 'y2': 'ymax'}, axis=1)
                rows = rows[['xmin', 'xmax', 'ymin', 'ymax']]
                recs = rows.to_dict(orient='records')

            llabs.append({'value': -1 if rows.shape[0] == 0 else 1, 
                            'id': i, 'dbidx': int(dbidx), 'boxes': recs})

        urls = get_image_paths(self.current_dataset, self.ev, next_idxs)
        pdata = {
            'image_urls': urls,
            'ldata': llabs,
            'refdata':reslabs,
        }
        return pdata


class ResetReq(BaseModel):
    dataset: str

class Box(BaseModel):
    xmin : float
    ymin : float
    xmax : float
    ymax : float

class LData(BaseModel):
    value : int
    id : int
    dbidx : int
    boxes : List[Box]

class NextReq(BaseModel):
    image_urls : List[str]
    ldata : List[LData]
    refdata : List[LData]

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

    @app.get('/getstate')
    def getstate(self):
        return self._getstate()

    @app.get('/datasets')
    def getdatasets(self):
        return self.datasets

    @app.post('/reset')
    def reset(self, r : ResetReq):
        print(f'resetting state with freshly constructed one for {r.dataset}')
        self._reset_dataset(r.dataset)
        return self._getstate()

    def _step(self, body: NextReq):
        if body is not None: ## refinement code
            box_dict = {}
            idxbatch = []
            for elt in body.ldata:

                df = pd.DataFrame([b.dict() for b in elt.boxes], 
                                    columns=['xmin', 'xmax', 'ymin', 'ymax']).astype('float32') # cols in case it is empty
                df = df.assign(dbidx=elt.dbidx)
                df = df.rename(mapper={'xmin':'x1', 'xmax':'x2', 'ymin':'y1', 'ymax':'y2'},axis=1)
                box_dict[elt.dbidx] = df
                idxbatch.append(elt.dbidx)

            self.state.ldata_db.update(box_dict)

            print('calling refine...')
            self.state.loop.refine(idxbatch=np.array(idxbatch), box_dict=box_dict)
            print('done refining...')

        self.state.step()
        return self.state.get_latest()

    @app.post('/next')
    def next(self, body : NextReq):
        return self._step(body)

    @app.post('/text')
    def text(self, key : str):
        self.state.loop.initialize(qstr=key) 
        return self._step(body=None)


# pylint: disable=maybe-no-member
WebSeesaw.deploy()
while True: # wait.
    input()