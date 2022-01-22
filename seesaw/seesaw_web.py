import json
import time
from typing import Optional, List, Dict
from fastapi.applications import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI

import os
from .dataset_manager import GlobalDataManager, IndexSpec
from .seesaw_bench import BenchParams
from .seesaw_session import Session, Box, SessionState, SessionParams


class AppState(BaseModel): # Using this as a response for every state transition.
    indices : List[IndexSpec]
    session : SessionState

class SessionReq(BaseModel):
    client_data : AppState

class ResetReq(BaseModel):
    index : IndexSpec

class GroundTruthReq(BaseModel):
    dataset_name : str
    ground_truth_category : str
    dbidx : int

class GroundTruthResp(BaseModel):
    dbidx : int
    boxes : List[Box]

class SessionInfoReq(BaseModel):
    path : str

class SaveResp(BaseModel):
    path : str

def prep_db(gdm, index_spec):
    hdb = gdm.load_index(index_spec.d_name, index_spec.i_name)
    # TODO: add subsetting here
    return hdb

def add_routes(app : FastAPI):
  class WebSeesaw:
      def __init__(self, root_dir, save_path):
          self.root_dir = root_dir
          self.save_path = save_path

          self.gdm = GlobalDataManager(root_dir)
          self.indices = self.gdm.list_indices()
          self.params = SessionParams(index_spec=self.indices[0],
                                    interactive='pytorch', 
                                    warm_start='warm', batch_size=3, 
                                    minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
                                    num_epochs=2, model_type='multirank2')

          self._reset_dataset(self.params.index_spec)

      def _reset_dataset(self, index_spec : IndexSpec):
          hdb = prep_db(self.gdm, index_spec)
          self.params.index_spec = index_spec
          self.session = Session(gdm=self.gdm, dataset=self.gdm.get_dataset(index_spec.d_name), hdb=hdb, params=self.params)

      def _getstate(self):
          return AppState(indices=self.indices, 
                              session=self.session.get_state())

      @app.get('/getstate', response_model=AppState)
      def getstate(self):
          return self._getstate()

      @app.post('/reset', response_model=AppState)
      def reset(self, r : ResetReq):
          print(f'resetting state with freshly constructed one for {r.index}')
          self._reset_dataset(r.index)
          return self._getstate()

      @app.post('/get_ground_truth', response_model=GroundTruthResp)
      def get_ground_truth(self, r : GroundTruthReq):
          pass
          # self.ground_truth_manager.get_ground_truth()

      @app.post('/next', response_model=AppState)
      def next(self, body : SessionReq):
          state = body.client_data.session
          if state is not None: ## refinement code
              self.session.update_state(state)
              self.session.refine()
          self.session.next()
          return self._getstate()

      @app.post('/text', response_model=AppState)
      def text(self, key : str):
          self.session.set_text(key=key)
          self.session.next()
          return self._getstate()

      @app.post('/save', response_model=SaveResp)
      def save(self, body : SessionReq):
          self.session.update_state(body.client_data.session)
          output_path = f'{self.save_path}/session_{time.strftime("%Y%m%d-%H%M%S")}'
          os.makedirs(output_path, exist_ok=False)
          base = self._getstate().dict()
          json.dump(base, open(f'{output_path}/summary.json', 'w'))
          return SaveResp(path=output_path)

      @app.post('/session_info', response_model=AppState)
      def session_info(self, body : SessionInfoReq):
          assert os.path.isdir(body.path)
          sum_path = f'{body.path}/summary.json'
          all_info  = json.load(open(sum_path, 'r'))
          if 'indices' not in all_info: # probably saved from benchmark
            all_info['indices'] = []
          return AppState(indices=all_info['indices'], session=all_info['session'])

  return WebSeesaw