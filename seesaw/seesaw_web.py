import json
import time
from typing import Optional, List, Dict
from fastapi.applications import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI

from .dataset_manager import GlobalDataManager, IndexSpec
from .seesaw_bench import BenchParams
from .seesaw_session import Session, Box, SessionState, SessionParams


class AppState(BaseModel): # Using this as a response for every state transition.
    indices : List[IndexSpec]
    current_index : IndexSpec
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

def add_routes(app : FastAPI):
  class WebSeesaw:
      def __init__(self, root_dir, save_path):
          self.root_dir = root_dir
          self.save_path = save_path

          self.gdm = GlobalDataManager(root_dir)
          self.indices = self.gdm.list_indices()
          self.current_index = self.indices[0]
          self.current_category = None
          self.bench_params = BenchParams(dataset_name=self.current_index.d_name,
                                          index_name=self.current_index.i_name, 
                                          ground_truth_category=self.current_category,
                                          qstr=None,
                                          n_batches=None, 
                                          max_feedback=None,
                                          box_drop_prob=None)

          self.params = SessionParams(interactive='pytorch', 
                                    warm_start='warm', batch_size=3, 
                                    minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
                                    tqdm_disabled=True, granularity='multi', positive_vector_type='vec_only', 
                                    num_epochs=2, n_augment=None, min_box_size=10, model_type='multirank2', 
                                    solver_opts={'C': 0.1, 'max_examples': 225, 'loss_margin': 0.05})

          self._reset_dataset(self.current_index)

      def _reset_dataset(self, index_spec : IndexSpec):
          hdb = self.gdm.load_index(index_spec.d_name, index_spec.i_name)
          self.session = Session(gdm=self.gdm, dataset=self.gdm.get_dataset(index_spec.d_name), hdb=hdb, params=self.params)
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

      @app.post('/save', response_model=AppState)
      def save(self, body : SessionReq):
          print('save req')
          self.session.update_state(body.client_data.session)
          output_path = f'{self.save_path}/session_summary_{time.strftime("%Y%m%d-%H%M%S")}'
          json.dump(self.bench_params, open(f'{output_path}/bench_params.json', 'w'))
          self.session.save_state(output_path)
          return self._getstate()

  return WebSeesaw