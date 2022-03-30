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
    session_id : str
    client_data : AppState

class ResetReq(BaseModel):
    session_id : str
    config : Optional[SessionParams]

class SessionInfoReq(BaseModel):
    path : str

class SaveResp(BaseModel):
    path : str

def prep_db(gdm, index_spec):
    hdb = gdm.load_index(index_spec.d_name, index_spec.i_name)
    # TODO: add subsetting here
    return hdb

from .util import reset_num_cpus
from .configs import _session_modes, _dataset_map, std_linear_config,std_textual_config

def session_params(session_mode, dataset_name, session_id, qkey, user):
  assert session_mode in _session_modes.keys()
  assert dataset_name in _dataset_map.keys()

  base = _session_modes[session_mode].copy(deep=True)
  base.index_spec.d_name = _dataset_map[dataset_name]
  ## base.index_spec.i_name set in template
  base.session_id = session_id
  base.other_params = {'mode':session_mode, 'dataset':dataset_name, 'qkey':qkey, 'user':user}
  return base

def add_routes(app : FastAPI):
  class WebSeesaw:
      session : Session
      sessions : Dict[str,Session]
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
          self.sessions = {}

      def _reset_dataset(self,  s: SessionParams):
          hdb = prep_db(self.gdm, s.index_spec)
          print('prep db done')
          res = make_session(self.gdm, s)
          session = res['session']
          self.sessions[s.session_id] = session
          self.session = session
          print('new session ready')

      def _getstate(self):
          return AppState(indices=self.indices, 
                          default_params=self.session.params if self.session is not None else self.default_params,
                          session=self.session.get_state() if self.session is not None else None)

      @app.get('/getstate', response_model=AppState)
      def getstate(self, session_id):
          self.session = self.sessions[session_id]
          return self._getstate()

      @app.post('/reset', response_model=AppState)
      def reset(self, r : ResetReq):
          print('reset request', r)
          if r.config is not None:
            self._reset_dataset(r.config)

          return self._getstate()

      @app.post('/next', response_model=AppState)
      def next(self, body : SessionReq):
          assert body.session_id in self.sessions
          self.session = self.sessions[body.session_id]
          state = body.client_data.session
          if state is not None: ## refinement code
              self.session.update_state(state)
              self.session.refine()
          self.session.next()
          return self._getstate()

      @app.post('/text', response_model=AppState)
      def text(self, session_id: str, key : str):
          assert session_id in self.sessions
          self.session = self.sessions[session_id]
          self.session.set_text(key=key)
          self.session.next()
          return self._getstate()

      @app.post('/save', response_model=SaveResp)
      def save(self, body : SessionReq):
          assert body.session_id in self.sessions
          self.session = self.sessions[body.session_id]
          self.session.update_state(body.client_data.session)
          self.session._log('save')
          output_path = f'{self.save_path}/session_{body.session_id}/session_time_{time.strftime("%Y%m%d-%H%M%S")}'
          os.makedirs(output_path, exist_ok=False)
          base = self._getstate().dict()
          json.dump(base, open(f'{output_path}/summary.json', 'w'))
          print(f'Saved session {output_path}')
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
      def user_session(self, mode, dataset, session_id, qkey, user):
        if session_id in self.sessions:
            print('using existing session')
            self.session = self.sessions[session_id]
        else:
            ## makes a new session using a config for the given mode
            print('start user_session request: ', mode, dataset, session_id)
            new_params = session_params(mode, dataset, session_id, qkey, user)
            print('new user_session params used:', new_params)
            self._reset_dataset(new_params)

        st =  self._getstate()
        print('completed user_session request')

        return st
  return WebSeesaw