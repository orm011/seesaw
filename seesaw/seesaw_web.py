import json
import time
from typing import Optional, List, Dict, Callable
from fastapi.applications import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, APIRouter
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from fastapi import FastAPI, Cookie
from fastapi import HTTPException

import os
from .dataset_manager import GlobalDataManager, IndexSpec
from .basic_types import Box, SessionState, SessionParams
from .seesaw_session import Session, make_session

from starlette.requests import Request
from starlette.responses import Response
import traceback
import sys


import pandas as pd


class TaskParams(BaseModel):
    session_id : str
    task_index : int
    qkey : str
    mode : str
    qstr : str
    dataset : str

class WorkerState(BaseModel):
    task_list : List[TaskParams]
    current_task_index : int

class AppState(BaseModel): # Using this as a response for every state transition.
    indices : Optional[List[IndexSpec]]
    worker_state : Optional[WorkerState]
    default_params : Optional[SessionParams]
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


import time

class WorkerState(BaseModel):
    task_list : List[TaskParams]
    current_task_index : int

class Worker:
    session_id : str
    task_list : List[TaskParams]
    current_task : int
    max_accepted : int 
    max_seen : int 
    max_time_s : float 

    def __init__(self, session_id, task_list, max_seen = 100, max_accepted = 10, max_time_s = pd.Timedelta('4 minutes').total_seconds()):
        self.session_id = session_id
        self.task_list = task_list
        self.current_task = -1

        self.max_seen = max_seen
        self.max_accepted = max_accepted
        self.max_time_s = max_time_s

    def get_state(self) ->WorkerState:
        return WorkerState(task_list=self.task_list, current_task=self.current_task)

    def is_session_done(self, session : Session) -> bool:
        tot = session.get_totals()
        start_time = session.action_log[0]['time']
        time_now = time.time()
        elapsed = time_now - start_time
        return tot['seen'] >= self.max_seen  or tot['accepted'] >= self.max_accepted or elapsed >= self.max_time_s

    def next_session_params(self):
        self.current_task += 1
        task = self.task_list[self.current_task]
        new_params = session_params(**task.dict())
        return new_params

class TaskInfoResp(BaseModel):
    session_id : str
    app_state : AppState

import random
import string
import traceback

def generate_id():
    return ''.join(random.choice(string.ascii_letters + '0123456789') for _ in range(32))

def prep_db(gdm, index_spec):
    hdb = gdm.load_index(index_spec.d_name, index_spec.i_name)
    # TODO: add subsetting here
    return hdb

from .util import reset_num_cpus
from .configs import _session_modes, _dataset_map, std_linear_config,std_textual_config

g_queries = {
    'pc':('bdd', 'police cars'), 
    'amb':('bdd', 'ambulances'),
    'dg':('bdd', 'dogs'), 
    'cd':('bdd', 'cars with open doors'), 
    'wch':('bdd', 'wheelchairs'),
    'mln':('coco', 'melons'),
    'spn':('coco', 'spoons'),
    'dst':('objectnet', 'dustpans'), 
    'gg':('objectnet', 'egg cartons'),
}

def session_params(mode, dataset, session_id, **kwargs):
  assert mode in _session_modes.keys()
  assert dataset in _dataset_map.keys()

  base = _session_modes[mode].copy(deep=True)
  base.index_spec.d_name = _dataset_map[dataset]
  ## base.index_spec.i_name set in template
  base.session_id = session_id
  base.other_params = {'mode':mode, 'dataset':dataset, **kwargs}
  return base

import random

def generate_task_list(mode, session_id):
    tasks = []
    # qs = random.shuffle(g_queries.items())
    # for q in :
    for i,k in enumerate(g_queries.keys()):
        (dataset, qstr) = g_queries[k]
        task = TaskParams(session_id=session_id, mode=mode, qkey=k, qstr=qstr, dataset=dataset, task_index=i)
        tasks.append(task)

    return tasks


# https://fastapi.tiangolo.com/advanced/custom-request-and-route/#accessing-the-request-body-in-an-exception-handler
class ErrorLoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            url = request.url._url
            cookies = request.cookies
            print(f'Received request: {url=} {cookies=}')
            try:
                ret = await original_route_handler(request)
            except Exception:
                (_,exc,_)=sys.exc_info()
                body = await request.body()
                req_body = body.decode()
                print(f'Exception {exc=} for request: {url=} {cookies=}\n{req_body=}')
                traceback.print_exc(file=sys.stdout)
                raise
            else:
              print(f'Successfully processed {url=} {cookies=} {ret=}')

            return ret

        return custom_route_handler


app = FastAPI()
app.router.route_class = ErrorLoggingRoute

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
        self.workers = {} # maps worker_id to state
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

    def _create_new_worker(self, mode):
        session_id = generate_id()
        self.workers[session_id] = Worker(session_id=session_id, task_list=generate_task_list(mode, session_id))
        return session_id

    def _reset_to_next_task(self, session_id):
        w = self.workers[session_id]
        new_params = w.next_session_params()
        self._reset_dataset(new_params)

    def _getstate(self):
        return AppState(#indices=self.indices, 
                        #default_params=self.session.params if self.session is not None else self.default_params,
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
        print('/next req for session ', body.session_id)
        assert body.session_id in self.sessions
        self.session = self.sessions[body.session_id]
        state = body.client_data.session
        if state is not None: ## refinement code
            self.session.update_state(state)
            self.session.refine()
        self.session.next()
        s =  self._getstate()
        print('done with /next req')
        return s

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
        
        qkey = self.session.params.other_params['qkey']
        if qkey not in g_queries:
            qkey = 'other'

        output_path = f'{self.save_path}/session_{body.session_id}/qkey_{qkey}/saved_{time.strftime("%Y%m%d-%H%M%S")}'
        #os.path.realpath(output_path)
        os.makedirs(output_path, exist_ok=False)
        base = self._getstate().dict()
        json.dump(base, open(f'{output_path}/summary.json', 'w'))
        print(f'Saved session {output_path}')
        return SaveResp(path='')

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
            new_params = session_params(mode, dataset, session_id, qkey=qkey, user=user)
            print('new user_session params used:', new_params)
            self._reset_dataset(new_params)

        st =  self._getstate()
        print('completed user_session request')
        return st
    
    @app.post('/session', response_model=AppState)
    def session(self, mode, session_id = Cookie(None)):
        """ assigns a new session id
        """
        if session_id is None:
            session_id = self._create_new_worker(mode)
            self._reset_to_next_task(session_id)
        elif session_id not in self.sessions:
            raise HTTPException(status_code=404, detail=f"unknown {session_id=}")
        else:
            pass

        self.session = self.sessions[session_id]
        st = self._getstate()
        if st.session.params.other_params['mode'] != mode:
            raise HTTPException(status_code=400, detail=f"session {session_id=} already exists with different mode")
        return st

