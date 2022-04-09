import json
import time
from typing import Optional, List, Dict, Callable
from fastapi.applications import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, APIRouter
from fastapi import Body, FastAPI, HTTPException
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

import time
import random
import string
import traceback
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

class Worker:
    session_id : str
    task_list : List[TaskParams]
    current_task : int

    def __init__(self, session_id, task_list):
        self.session_id = session_id
        self.task_list = task_list
        self.current_task = -1

    def get_state(self) -> WorkerState:
        return WorkerState(task_list=self.task_list, current_task_index=self.current_task)

    def next_session(self):
        self.current_task += 1
        task = self.task_list[self.current_task]
        new_params = session_params(**task.dict())
        return new_params

def generate_id():
    return ''.join(random.choice(string.ascii_letters + '0123456789') for _ in range(32))

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

class SingleSeesaw:
    """ holds the state for a single user state machine. All actions here are serially run.
        API mirrors the one in WebSeesaw for single user operations
    """
    session_id : str
    session : Optional[Session]
    worker : Optional[Worker]

    def __init__(self, root_dir, save_path, session_id, worker : Worker = None, num_cpus=None):
        if num_cpus is not None:
            reset_num_cpus(num_cpus)

        self.session_id = session_id
        self.root_dir = root_dir
        self.save_path = save_path
        self.worker = worker

        self.gdm = GlobalDataManager(root_dir)
        self.indices = self.gdm.list_indices()
        self.session = None

    def _reset_dataset(self,  s: SessionParams):
        res = make_session(self.gdm, s)
        self.session = res['session']

    def next_task(self):
        params = self.worker.next_session()
        return self._reset_dataset(params)

    def getstate(self):
        return  AppState(indices=None,
                        default_params=None,
                        worker_state=self.worker.get_state() if self.worker else None,
                        session=self.session.get_state() if self.session else None)

    def reset(self, r : ResetReq):
        if r.config is not None:
            self._reset_dataset(r.config)
        return self.getstate()

    def next(self, body : SessionReq):
        state = body.client_data.session
        if state is not None: ## refinement code
            self.session.update_state(state)
            self.session.refine()
        self.session.next()
        return self.getstate()

    def text(self, key : str):
        self.session.set_text(key=key)
        self.session.next()
        return self.getstate()

    def save(self, body : SessionReq):
        if body.client_data.session:
            self.session.update_state(body.client_data.session)

        self.session._log('save')        
        qkey = self.session.params.other_params['qkey']
        if qkey not in g_queries:
            qkey = 'other'
        
        output_path = f'{self.save_path}/session_{self.session_id}/qkey_{qkey}/saved_{time.strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(output_path, exist_ok=False)
        base = self.getstate().dict()
        json.dump(base, open(f'{output_path}/summary.json', 'w'))
        print(f'saved session {output_path}')
        return SaveResp(path='')

class WebSeesaw:
    sessions : Dict[str,SingleSeesaw]
    def __init__(self, root_dir, save_path, num_cpus=None):
        if num_cpus is not None:
            reset_num_cpus(num_cpus)

        self.root_dir = root_dir
        self.save_path = save_path
        self.num_cpus = num_cpus
        self.sessions = {} # maps session id to state...

    def _create_new_worker(self, mode):
        session_id = generate_id()
        worker = Worker(session_id=session_id, task_list=generate_task_list(mode, session_id))
        self.sessions[session_id] = SingleSeesaw(self.root_dir, self.save_path, session_id, worker, num_cpus=self.num_cpus)
        return session_id

    @app.post('/user_session', response_model=AppState)
    def user_session(self, mode, dataset, session_id, qkey, user):
        """ API for the old-school user study where we generated URLs and handed them out.
        """ 
        # will make a new session if the id is new
        if session_id not in self.sessions:
            ## makes a new session using a config for the given mode
            print('start user_session request: ', mode, dataset, session_id)
            new_params = session_params(mode, dataset, session_id, qkey=qkey, user=user)
            print('new user_session params used:', new_params)
            new_session = SingleSeesaw(self.root_dir, self.save_path, session_id=session_id, num_cpus=self.num_cpus)
            new_session._reset_dataset(new_params)
            self.sessions[session_id] = new_session

        return self.sessions[session_id].getstate()
    
    @app.post('/session', response_model=AppState)
    def session(self, mode : str, response : Response, session_id = Cookie(None)):
        """ creates a new (multi-session) session_id  and sets a session id cookie when none exists.
        """
        if session_id is None:
            session_id = self._create_new_worker(mode)
            response.set_cookie(key='session_id', value=session_id, max_age=pd.Timedelta('2 hours').total_seconds())
        elif session_id not in self.sessions:
            raise HTTPException(status_code=404, detail=f"unknown {session_id=}")
        else:
            pass

        s = self.sessions[session_id]
        st = s.getstate()
        if st.session:
            if st.session.params.other_params['mode'] != mode:
                raise HTTPException(status_code=400, detail=f"session {session_id=} already exists with different mode")
        
        return st

    @app.post('/session_info', response_model=AppState)
    def session_info(self, body : SessionInfoReq):
        """ Used for visualizing (read-only) old session logs stored in files
        """
        assert os.path.isdir(body.path)
        sum_path = f'{body.path}/summary.json'
        all_info  = json.load(open(sum_path, 'r'))
        if 'bench_params' in all_info: # saved benchmark
            return AppState(indices=None, worker_state=None,
                            session=all_info['result']['session'], 
                            default_params=all_info['result']['session']['params'])

        else: # saved web session
            return AppState(indices=None, worker_state=None,
                            session=all_info['session'], 
                            default_params=all_info['session']['params'])

    
    """
        Single-session forwarding functions (forward calls)
    """
    @app.get('/getstate', response_model=AppState)
    def getstate(self, session_id = Cookie(None)):
        return self.sessions[session_id].getstate()

    @app.post('/reset', response_model=AppState)
    def reset(self, r : ResetReq, session_id = Cookie(None)):
        return self.sessions[session_id].reset(r)

    @app.post('/next', response_model=AppState)
    def next(self, body : SessionReq, session_id = Cookie(None)):
        return self.sessions[session_id].next(body)

    @app.post('/text', response_model=AppState)
    def text(self, key : str, session_id = Cookie(None)):
        return self.sessions[session_id].text(key)

    @app.post('/next_task', response_model=AppState)
    def next_task(self, session_id = Cookie(None)):
        return self.sessions[session_id].next_task()

    @app.post('/save', response_model=SaveResp)
    def save(self, body : SessionReq, session_id = Cookie(None)):
        return self.sessions[session_id].save(body)
