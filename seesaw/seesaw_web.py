import json
import time
from typing import Optional, List, Dict, Callable
from fastapi.applications import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, APIRouter
from fastapi import Body, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from fastapi import FastAPI, Cookie, Depends
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

import ray
import ray.serve


class TaskParams(BaseModel):
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

class SearchDesc(BaseModel):
    dataset : str
    qstr : str
    description : str = ''

class NotificationState(BaseModel): 
    urls : List[str]
    description : str

class SessionReq(BaseModel):
    client_data : AppState

class ResetReq(BaseModel):
    config : Optional[SessionParams]

class SessionInfoReq(BaseModel):
    path : str

class SaveResp(BaseModel):
    path : str

class EndSession(BaseModel):
    token : str

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
    'pc':SearchDesc(dataset='bdd', qstr='police cars', 
                description='''Police vehicles that have lights and some marking related to police. 
                    Sometimes private security vehicles or ambulances look like police cars but should not be included'''),
    'dg':SearchDesc(dataset='bdd', qstr='dogs'),
    'cd':SearchDesc(dataset='bdd', qstr='car with open doors', 
                description='''Any vehicles with any open doors, including open trunks in cars, and rolled-up doors in trucks and trailers.
                         When there are too many vehicles in a congested street, only focus on the foreground. 
                         These can be rare, so you need to be very careful not to miss them'''),
    'wch':SearchDesc(dataset='bdd', qstr='wheelchairs',
                description='''We include wheelchair alternatives such as electric scooters for the mobility impaired. 
                                We do not include the wheelchair icon (♿), or baby strollers'''),
    'mln':SearchDesc(dataset='coco', qstr='cantaloupe or honeydew melon', 
                description='''We inclulde both cantaloupe (orange melon) and honeydew (green melon), whole melons and melon pieces. 
                                We dont include any other types of melon, including watermelons, papaya or pumpkins, which can look similar. 
                                If you cannot tell whether a fruit piece is really from melon just leave it out.'''),
    'spn':SearchDesc(dataset='coco', qstr='spoons or teaspoons', 
                description='''We include spoons or teaspons of any material for eating. 
                    We dont include the large cooking or serving spoons, ladles for soup, or measuring spoons.'''),
    'dst':SearchDesc(dataset='objectnet', qstr='dustpans',
                description='''We include dustpans on their own or together with other tools, like brooms, from any angle.'''),
    'gg':SearchDesc(dataset='objectnet', qstr='egg cartons',
                description='''These are often made of cardboard or styrofoam. We include them viewed from any angle. 
                            We dont include the permanent egg containers that come in the fridge''')
}

start_url = '/home/gridsan/groups/fastai/seesaw/data/'

def session_params(mode, dataset, **kwargs):
  assert mode in _session_modes.keys()
  assert dataset in _dataset_map.keys()

  base = _session_modes[mode].copy(deep=True)
  base.index_spec.d_name = _dataset_map[dataset]
  ## base.index_spec.i_name set in template
  base.other_params = {'mode':mode, 'dataset':dataset, **kwargs}
  return base


def generate_task_list(mode):
    tasks = []
    # qs = random.shuffle(g_queries.items())
    # for q in :
    for i,k in enumerate(g_queries.keys()):
        sdesc = g_queries[k]
        task = TaskParams(mode=mode, qkey=k, qstr=sdesc.qstr, dataset=sdesc.dataset, task_index=i)
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

class WebSession:
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

    def next_task(self, body : SessionReq):
        if self.session: # null the first time
            self.session._log('next_task')
            self.save(body)

        params = self.worker.next_session()
        self._reset_dataset(params)
        return self.getstate()

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
        self.save(body)
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

    def save(self, body : SessionReq = None):
        if self.session is not None:
            if body and body.client_data and body.client_data.session:
                self.session.update_state(body.client_data.session)

            self.session._log('save')        
            qkey = self.session.params.other_params.get('qkey', None)

            # ensure session id is set correctly in json for easier access at read time
            self.session.params.other_params['session_id'] = self.session_id
            save_time = time.strftime("%Y%m%d-%H%M%S")
            self.session.params.other_params['save_time'] = save_time

            if qkey not in g_queries:
                qkey = 'other'
            
            output_path = f'{self.save_path}/session_{self.session_id}/qkey_{qkey}/saved_{save_time}'
            os.makedirs(output_path, exist_ok=True)
            base = self.getstate().dict()
            json.dump(base, open(f'{output_path}/summary.json', 'w'))
            print(f'saved session {output_path}')
            return SaveResp(path='')

    def sleep(self):
        start = time.time()
        time.sleep(10)
        end = time.time()
        return end - start

    def test(self):
        return True


WebSessionActor = ray.remote(WebSession)

from ray.actor import ActorHandle

class SessionManager:
    sessions : Dict[str, ActorHandle]

    def __init__(self, root_dir, save_path, num_cpus_per_session):
        self.root_dir = root_dir
        self.save_path = save_path
        self.num_cpus = num_cpus_per_session
        self.sessions = {}


    def _new_session(self, task_list):
        session_id = generate_id()
        worker = Worker(session_id=session_id, task_list=task_list)
        self.sessions[session_id] = WebSessionActor.options(name=f'web_session#{session_id}', 
                                    num_cpus=self.num_cpus).remote(self.root_dir, self.save_path, 
                                                session_id, worker, 
                                                num_cpus=self.num_cpus)
        return session_id
    
    def new_worker(self, mode):
        task_list = generate_task_list(mode)
        return self._new_session(task_list)

    def new_session(self):
        return self._new_session([])

    def session_exists(self, session_id):
        return session_id in self.sessions

    def end_session(self, session_id):
        ## session should die after reference 
        sess = self.sessions[session_id]
        del self.sessions[session_id]
        print(f'ending session {session_id}')
        ray.kill(sess)

    def get_session(self, session_id):
        return self.sessions.get(session_id)


async def get_handle(session_id : str = Cookie(None)) -> ActorHandle:
    """ common code to get the handle for a running session, shared across several calls
    """
    if session_id is None:
        raise HTTPException(status_code=404, detail=f"this API requires a session_id")

    session_manager = ray.get_actor('session_manager')
    handle = await session_manager.get_session.remote(session_id)
    
    if handle is None:
        raise HTTPException(status_code=404, detail=f"unknown {session_id=}")

    return handle

SessionManagerActor = ray.remote(SessionManager)

@ray.serve.deployment(name="seesaw_deployment", num_replicas=1, route_prefix='/')
@ray.serve.ingress(app)
class WebSeesaw:
    """
    when exposed by ray serve/uvicorn, the code below
    (seems to) run as multi-threaded python (ie multiple requests will be served in simultaneously, at least that seemed to
    be the case with a time.sleep() call, and will see partially modified state in the class). 
    
    I'm not sure what assumptions the serve framework / fastapi 
    make of this class, so I'm avoiding shared state in this class (eg, no session dictionary here), Instead,
    we use a separate actor for mapping sessions / adding sessions
    and separately we also use one actor per session since each of them needs its own resources (eg cpu access etc).
    """
    session_manager : ActorHandle

    def __init__(self, session_manager):
        print('WebSeesaw init method called')
        self.session_manager = session_manager

    @app.post('/user_session', response_model=AppState)
    async def user_session(self, mode, dataset, qkey, user, response : Response, session_id = Cookie(None)):
        """ API for the old-school user study where we generated URLs and handed them out.
        """ 
        new_session = False
        if session_id is None:
            session_id = await self.session_manager.new_session.remote()
            response.set_cookie(key='session_id', value=session_id, max_age=pd.Timedelta('2 hours').total_seconds())
            new_session = True
            
        handle = await get_handle(session_id)
        if new_session:
            new_params = session_params(mode, dataset, qkey=qkey, user=user)
            await handle._reset_dataset.remote(new_params)

        return await handle.getstate.remote()
    
    @app.post('/session', response_model=AppState)
    async def session(self, mode : str, response : Response, session_id = Cookie(None)):
        """ creates a new (multi-session) session_id  and sets a session id cookie when none exists.
        """
        if session_id is None:
            session_id = await self.session_manager.new_worker.remote(mode)
            response.set_cookie(key='session_id', value=session_id, max_age=pd.Timedelta('2 hours').total_seconds())                

        handle = await get_handle(session_id)
                
        st = await handle.getstate.remote()
        if st.session:
            real_mode = st.session.params.other_params['mode']
            if real_mode != mode:
                raise HTTPException(status_code=400, detail=f"{session_id=} already exists with {real_mode=}")
        
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

    @app.get('/task_description', response_model=NotificationState)
    def task_description(self, code : str):
        sdesc = g_queries[code] 

        description = f"""In the following task, you'll be looking for {sdesc.qstr}. 
                            {sdesc.description}.
                            Below are some examples of {sdesc.qstr}. When you are ready and the OK button is enabled, press it to proceed."""
        urls = []
        for i in range(4): 
            url = start_url + '/examples/' + code + '/' + code + '-' + str(i+1) + '.png'
            urls.append(url)

        return NotificationState(
            description = description, 
            urls = urls, 
        )

    @app.post('/session_end', response_model=EndSession)
    async def session_end(self, response : Response, session_id = Cookie(None),  body : SessionReq = None):

        # no matter what, expire cookie from requester
        response.set_cookie('session_id', max_age=0)

        sess_exists = await self.session_manager.session_exists.remote(session_id)        
        if not sess_exists: # after restarting server there are old ids out there that don't exist
            return EndSession(token=session_id)
        
        handle = await get_handle(session_id)
        await handle.save.remote(body)
        await self.session_manager.end_session.remote(session_id)
        return EndSession(token=session_id)

    """
        Single-session forwarding functions (forward calls)
    """
    @app.get('/getstate', response_model=AppState)
    async def getstate(self, handle=Depends(get_handle)):
        return await handle.getstate.remote()

    @app.post('/reset', response_model=AppState)
    async def reset(self, r : ResetReq, handle=Depends(get_handle)):
        return await handle.reset.remote(r)

    @app.post('/next', response_model=AppState)
    async def next(self, body : SessionReq, handle=Depends(get_handle)):
        return await handle.next.remote(body)

    @app.post('/text', response_model=AppState)
    async def text(self, key : str, handle=Depends(get_handle)):
        return await handle.text.remote(key)

    @app.post('/save', response_model=SaveResp)
    async def save(self, body : SessionReq, handle=Depends(get_handle)):
        return await handle.save.remote(body)

    @app.post('/next_task', response_model=AppState)
    async def next_task(self, body : SessionReq, handle=Depends(get_handle)):
        return await handle.next_task.remote(body)

    @app.post('/sleep')
    async def sleep(self, handle=Depends(get_handle)):
        return await handle.sleep.remote()

    @app.post('/test')
    async def test(self, handle=Depends(get_handle)):
        return await handle.test.remote()