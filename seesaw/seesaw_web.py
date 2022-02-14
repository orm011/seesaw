import json
import time
from typing import Optional, List, Dict
from fastapi.applications import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI

import os
from .dataset_manager import GlobalDataManager, IndexSpec
from .basic_types import Box, SessionState, SessionParams
from .seesaw_session import Session, make_session

class AppState(BaseModel): # Using this as a response for every state transition.
    indices : List[IndexSpec]
    default_params : SessionParams
    session : Optional[SessionState] #sometimes there is no active session

class SessionReq(BaseModel):
    client_data : AppState

class ResetReq(BaseModel):
    config : Optional[SessionParams]

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
    print('call load index')
    hdb = gdm.load_index(index_spec.d_name, index_spec.i_name)
    # TODO: add subsetting here
    return hdb

from .textual_feedback_box import std_textual_config
from .util import reset_num_cpus


_session_modes = {
  'default':SessionParams(index_spec={'d_name':'', 'i_name':''},
          interactive='plain', 
          method_config=None,
          warm_start='warm', batch_size=3, 
          minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
          num_epochs=2, model_type='cosine'),
  'box':SessionParams(index_spec={'d_name':'', 'i_name':''},
          interactive='pytorch', 
          method_config=None,
          warm_start='warm', batch_size=3, 
          minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
          num_epochs=2, model_type='cosine'),
  'textual':SessionParams(index_spec={'d_name':'', 'i_name':''},
          interactive='textual', 
          method_config=std_textual_config,
          warm_start='warm', batch_size=3, 
          minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
          num_epochs=2, model_type='cosine'),
}

_dataset_map = {
  'lvis':'data/lvis/',
  'coco':'data/coco/',
  'bdd':'data/bdd/',
  'objectnet': 'data/objectnet/'
}

def session_params(session_mode, dataset_name):
  assert session_mode in _session_modes.keys()
  assert dataset_name in _dataset_map.keys()

  base = _session_modes[session_mode].copy(deep=True)
  base.index_spec.d_name = _dataset_map[dataset_name]
  base.index_spec.i_name = 'coarse' if session_mode in ['default'] else 'multiscale'
  return base

def add_routes(app : FastAPI):
  class WebSeesaw:
      def __init__(self, root_dir, save_path, num_cpus=None):
          if num_cpus is not None:
            reset_num_cpus(num_cpus)

          self.root_dir = root_dir
          self.save_path = save_path

          self.gdm = GlobalDataManager(root_dir)
          print('gdm done')
          self.indices = self.gdm.list_indices()
          print(self.indices)
          print('indices done')
          self.default_params = _session_modes['textual']
          self.session = None


      def _reset_dataset(self,  s: SessionParams):
          hdb = prep_db(self.gdm, s.index_spec)
          print('prep db done')
          res = make_session(self.gdm, s)
          self.session = res['session']
          print('new session ready')

      def _getstate(self):
          return AppState(indices=self.indices, 
                          default_params=self.session.params if self.session is not None else self.default_params,
                          session=self.session.get_state() if self.session is not None else None)

      @app.get('/getstate', response_model=AppState)
      def getstate(self):
          return self._getstate()

      @app.post('/reset', response_model=AppState)
      def reset(self, r : ResetReq):
          print('reset request', r)
          if r.config is None:
            self.session = None
          else:
            self._reset_dataset(r.config)

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
          if 'bench_params' in all_info: # saved benchmark
            return AppState(indices=[], session=all_info['result']['session'], default_params=all_info['result']['session']['params'])
          else: # saved web session
            return AppState(indices=all_info['indices'], session=all_info['session'], default_params=all_info['session']['params'])

      @app.post('/user_session', response_model=AppState)
      def user_session(self, mode, dataset):
        ## makes a new session using a config for the given mode
        new_params = session_params(mode, dataset)
        self._reset_dataset(new_params)
        return self._getstate()

  return WebSeesaw