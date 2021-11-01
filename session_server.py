import ray
import fastapi
from fastapi import FastAPI

import typing
import pydantic
from typing import Optional
from pydantic import BaseModel
import ray.serve
from ray import serve

app = FastAPI()

print('init ray...')
ray.init('auto', namespace="seesaw")
print('inited.')

print('start serve...')
ray.serve.start() ## will use localhost:8000. the documented ways of specifying this are currently failing...
print('started.')

print('importing seesaw...')
import numpy as np
from seesaw import EvDataset, SeesawLoop, LoopParams, ModelService, GlobalDataManager
print('imported.')

def get_image_paths(dataset_name, ev, idxs):
    return [ f'/data/{dataset_name}/images/{ev.image_dataset.paths[int(i)]}' for i in idxs]

class SessionState:
    current_dataset : str
    ev : EvDataset
    loop : SeesawLoop
    acc_indices : list = [np.zeros(0, dtype=np.int32)]

    def __init__(self, dataset_name, ev):
        self.current_dataset = dataset_name
        self.ev = ev

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
        return dat

    def get_latest(self):
        dat = self.get_panel_data(next_idxs=self.acc_indices[-1])
        return dat

    def get_panel_data(self, *, next_idxs, label_db=None):
        reslabs = []
        print(next_idxs)
        for (i,dbidx) in enumerate(next_idxs):
            # boxes = copy.deepcopy(label_db.get(dbidx, None))
            boxes = []
            print(dbidx)
            reslabs.append({'value': -1 if boxes is None else 1 if len(boxes) > 0 else 0, 
                            'id': i, 'dbidx': int(dbidx), 'boxes': boxes})
        urls = get_image_paths(self.current_dataset, self.ev, next_idxs)
        pdata = {
            'image_urls': urls,
            'ldata': reslabs,
        }
        return pdata


class ResetReq(BaseModel):
    dataset: str

@serve.deployment(name="seesaw_deployment", route_prefix='/')
@serve.ingress(app)
class WebSeesaw:
    def __init__(self):
        self.gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/data')
        self.datasets = ['objectnet', 'dota', 'lvis','coco', 'bdd']
        ## initialize to first one
        self.xclip = ModelService(ray.get_actor('clip#actor'))
        self._reset_dataset(self.datasets[0])

    def _get_ev(self, dataset_name):
        actor = ray.get_actor(f'{dataset_name}#actor')
        ev = ray.get(ray.get(actor.get_ev.remote()))
        return ev

    def _reset_dataset(self, dataset_name):
        ev = self._get_ev(dataset_name)
        self.state = SessionState(dataset_name, ev)

    @app.get('/getstate')
    def getstate(self):
        s = self.state.get_state()
        s['datasets'] = self.datasets
        print(s)
        return s

    @app.get('/datasets')
    def getdatasets(self):
        return self.datasets

    @app.post('/reset')
    def reset(self, r : ResetReq):
        print(f'resetting state with freshly constructed one for {r.dataset}')
        self._reset_dataset(r.dataset)
        return self.getstate()

    @app.post('/next')
    def step(self):
        if False: ## refinement code
            pass
            # ldata = request.json.get('ldata', [])
            # #update_db(state.bfq.label_db, ldata)
            # indices = np.array([litem['dbidx'] for litem in ldata])        
        self.state.step()
        return self.state.get_latest()

    @app.post('/text')
    def text(self, key : str):
        self.state.loop.initialize(qstr=key) 
        return self.step()


# pylint: disable=maybe-no-member
WebSeesaw.deploy()

while True:
    input()