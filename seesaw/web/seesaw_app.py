import json
from seesaw.dataset_manager import GlobalDataManager
import time
from typing import Optional, List, Dict, Callable
from fastapi.applications import FastAPI
from fastapi import FastAPI, APIRouter
from fastapi import Body, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from fastapi import FastAPI, Cookie, Depends
from fastapi import HTTPException

import os

from starlette.requests import Request
from starlette.responses import Response
import traceback
import sys

import time
import traceback
import pandas as pd

import glob
import ray

from .common import *
from .session_manager import get_manager

from ray.actor import ActorHandle


# https://fastapi.tiangolo.com/advanced/custom-request-and-route/#accessing-the-request-body-in-an-exception-handler
class ErrorLoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            url = request.url._url
            cookies = request.cookies
            print(f"Received request: {url=} {cookies=}")
            try:
                ret = await original_route_handler(request)
            except Exception:
                (_, exc, _) = sys.exc_info()
                body = await request.body()
                req_body = body.decode()
                print(f"Exception {exc=} for request: {url=} {cookies=}\n{req_body=}")
                traceback.print_exc(file=sys.stdout)
                raise
            else:
                print(f"Successfully processed {url=} {cookies=} {ret=}")

            return ret

        return custom_route_handler


async def get_handle(session_id: str = Cookie(None)) -> ActorHandle:
    """common code to get the handle for a running session, shared across several calls"""
    if session_id is None:
        raise HTTPException(status_code=404, detail=f"this API requires a session_id")

    session_manager = ray.get_actor("session_manager")
    handle = await session_manager.get_session.remote(session_id)

    if handle is None:
        raise HTTPException(status_code=404, detail=f"unknown {session_id=}")

    return handle


start_url = "/home/gridsan/groups/fastai/seesaw/data/examples2/"
urls = {}
neg_urls = {}
for key in g_queries.keys():
    urls[key] = []
    neg_urls[key] = []
    root = start_url + key + "/"
    # glob.glob(‘path/**jpg’) + glob.glob(‘path/**webp’) + glob.glob(’path/**png)
    paths = (
        glob.glob(root + "**jpg")
        + glob.glob(root + "**webp")
        + glob.glob(root + "**png")
        + glob.glob(root + "/neg/" + "**jpg")
        + glob.glob(root + "/neg/" + "**webp")
        + glob.glob(root + "/neg/" + "**png")
    )
    for path in paths:
        if path.find("/neg") != -1:  # Negative
            neg_urls[key].append(path)
        else:
            urls[key].append(path)


app = FastAPI()
app.router.route_class = ErrorLoggingRoute

from seesaw.configs import get_session_params_from_yaml

@app.post("/user_session", response_model=AppState)
async def user_session(
    mode : str,
    dataset : str,
    index : str, 
    #qkey,
    #user,
    annotation_category : str,
    response: Response,
    session_id=Cookie(default=None),
    manager=Depends(get_manager),
):
    """API for the old-school user study where we generated URLs and handed them out."""
    new_session = False
    if session_id is None:
        session_id = await manager.new_session.remote()
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=pd.Timedelta("2 hours").total_seconds(),
        )
        new_session = True
    handle = await get_handle(session_id)
    if new_session:
        if mode.startswith('yaml_'):
            print('reading session params from file')
            config_name = mode[len('yaml_'):]
            new_params = get_session_params_from_yaml(config_name, dataset, index, annotation_category)
            print(new_params)
        else:
            new_params = session_params(mode, dataset, index)
        await handle._reset_dataset.remote(new_params)

    return await handle.getstate.remote()

@app.post("/session", response_model=AppState)
async def session(
    mode: str,
    response: Response,
    manager=Depends(get_manager),
    session_id=Cookie(None),
):
    """creates a new (multi-session) session_id  and sets a session id cookie when none exists."""
    if session_id is None:
        session_id = await manager.new_worker.remote(mode)
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=pd.Timedelta("2 hours").total_seconds(),
        )

    handle = await get_handle(session_id)

    st = await handle.getstate.remote()
    if st.session:
        real_mode = st.session.params.other_params["mode"]
        if real_mode != mode:
            raise HTTPException(
                status_code=400,
                detail=f"{session_id=} already exists with {real_mode=}",
            )

    return st

import random
import string

## 3 things:
# a save path that is related is available
# data is saved on session end and can be read again 

from seesaw.labeldb import LabelDB
from seesaw.basic_types import Imdata

def get_image_reference_data(dataset, *, annotation_category, idxbatch):
    reslabs = []
    bd, _ = dataset.load_ground_truth()
    bd = bd[bd.category == annotation_category]

    if idxbatch is None or idxbatch == []:
        idxbatch = bd.dbidx.unique()

    label_db = LabelDB()
    label_db.fill(bd)
    urls = dataset.get_urls(idxbatch)

    for i, (url, dbidx) in enumerate(zip(urls, idxbatch)):
        dbidx = int(dbidx)
        boxes = label_db.get(dbidx, format="box")

        elt = Imdata(
            url=url,
            dbidx=dbidx,
            boxes=boxes,
            activations=None,
            timing=[]
        )
        reslabs.append(elt)
    return reslabs

from seesaw.basic_types import SessionState, SessionParams, IndexSpec
import yaml

@app.post("/annotate", response_model=AppState)
async def annotate(dataset : str, category : str, pathfile : str):
    """show current annotations for paths and allow saving edited"""

    save_path = f"{pathfile}/summary.json"
    yamlfile = f"{pathfile}/paths.yaml" # explicit list
    assert not os.path.exists(save_path) # don't allow overwriting
    pathfile = pathfile.rstrip('/')

    gdm = GlobalDataManager('/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/')
    ds = gdm.get_dataset(dataset)

    if os.path.exists(yamlfile):
        paths = yaml.safe_load(open(yamlfile, 'r'))
        idxs = []
        path2dbidx = {path:i for (i,path) in enumerate(ds.paths)}
        for p in paths:
            dbidx = path2dbidx.get(p, -1)
            assert dbidx != -1
            idxs.append(dbidx)
    else: # figure out idxbatch based on category
        idxs = []

    ## make a dummy session object compatible with the frontend
    params = SessionParams(annotation_category=category, 
                    interactive='dummy',
                    batch_size=len(idxs), # dummy
                    index_spec=IndexSpec(d_name=dataset, 
                            i_name='multiscalecoarse',  # dummy
                            c_name=None))           
    ## now pre-fill it with current annotations
    gdata = get_image_reference_data(ds, annotation_category=category, idxbatch=idxs)
    session = SessionState(params=params, gdata=[gdata], timing=[], reference_categories=[])

    return AppState(
            indices=None,
            worker_state=None,
            session=session,
            default_params=session.params,
            save_path=save_path
        )


@app.post("/session_info", response_model=AppState)
async def session_info(path : str,
                annotation_category: str = None):
    """Used for visualizing of session logs stored in files"""
    assert os.path.isdir(path)
    path = path.rstrip('/')
    sum_path = f"{path}/summary.json"
    all_info = json.load(open(sum_path, "r"))
    random_id = ''.join([random.choice(string.ascii_lowercase) for i in range(10)])
    if annotation_category is not None:
        save_path = f'{path}_annot_{random_id}'
        assert not os.path.exists(save_path)
        print(f'using {save_path=}')
    else:
        save_path = None

    if "bench_params" in all_info:  # saved benchmark
        session = all_info["result"]["session"]
    else: # web session
        session = all_info["session"]

    if annotation_category is not None: # set this 
        session['params']['annotation_category'] = annotation_category
        ## now pre-fill it with current annotations
        gdm = GlobalDataManager('/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/')
        dataset_name = session['params']['index_spec']['d_name']
        ds = gdm.get_dataset(dataset_name)
        idxs = []
        for r in session['gdata']:
            for elt in r:
                idxs.append(int(elt['dbidx']))

        new_gdata = get_image_reference_data(ds, annotation_category=annotation_category, idxbatch=idxs)
        session['gdata'] = [new_gdata]

    return AppState(
            indices=None,
            worker_state=None,
            session=session,
            default_params=session["params"],
            save_path=path
        )


@app.get("/task_description", response_model=NotificationState)
def task_description(code: str):
    sdesc = g_queries[code]
    # description = f"""In the following task, you'll be looking for {sdesc.qstr}.
    #                     {sdesc.description}.
    #                     Below are some examples of {sdesc.qstr}. When you are ready and the OK button is enabled, press it to proceed."""

    return NotificationState(
        description=sdesc,
        urls=urls[code],
        neg_urls=neg_urls[code],
    )

@app.post("/session_end", response_model=EndSession)
async def session_end(
    response: Response,
    session_id=Cookie(None),
    manager=Depends(get_manager),
    body: SessionReq = None,
):
    # no matter what, expire cookie from requester
    if session_id is not None:
        response.set_cookie("session_id", max_age=0)
        sess_exists = await manager.session_exists.remote(session_id)
        if (
            not sess_exists
        ):  # after restarting server there are old ids out there that don't exist
            return EndSession(token=session_id)

        handle = await get_handle(session_id)
        await handle.save.remote(body)
        await manager.end_session.remote(session_id)
        return EndSession(token=session_id)
    elif body is None:
        print(' empty body and no session id. doing nothing')
        return EndSession(token=None)
    else:
        app_state = body.client_data
        save_path = app_state.save_path
        if save_path is None:
            print('session has no savepath, doing nothing')
            return EndSession(token=None)

        os.makedirs(save_path, exist_ok=True)
        json.dump(app_state.dict(), open(f"{save_path}/summary.json", "w"))
        print(f"saved session {save_path}")
        return EndSession(token=None)

"""
    Single-session forwarding functions (forward calls)
"""


@app.get("/getstate", response_model=AppState)
async def getstate(handle=Depends(get_handle)):
    return await handle.getstate.remote()


@app.post("/reset", response_model=AppState)
async def reset(r: ResetReq, handle=Depends(get_handle)):
    return await handle.reset.remote(r)


@app.post("/next", response_model=AppState)
async def next(body: SessionReq, handle=Depends(get_handle)):
    return await handle.next.remote(body)


@app.post("/text", response_model=AppState)
async def text(key: str, handle=Depends(get_handle)):
    return await handle.text.remote(key)


@app.post("/save", response_model=SaveResp)
async def save(body: SessionReq, handle=Depends(get_handle)):
    return await handle.save.remote(body)


@app.post("/next_task", response_model=AppState)
async def next_task(body: SessionReq, handle=Depends(get_handle)):
    return await handle.next_task.remote(body)


@app.post("/sleep")
async def sleep(handle=Depends(get_handle)):
    return await handle.sleep.remote()


@app.post("/test")
async def test(handle=Depends(get_handle)):
    return await handle.test.remote()
