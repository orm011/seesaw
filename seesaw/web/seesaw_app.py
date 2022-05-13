import json
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


@app.post("/user_session", response_model=AppState)
async def user_session(
    mode,
    dataset,
    qkey,
    user,
    response: Response,
    session_id=Cookie(None),
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
        new_params = session_params(mode, dataset, qkey=qkey, user=user)
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


@app.post("/session_info", response_model=AppState)
def session_info(body: SessionInfoReq):
    """Used for visualizing (read-only) old session logs stored in files"""
    assert os.path.isdir(body.path)
    sum_path = f"{body.path}/summary.json"
    all_info = json.load(open(sum_path, "r"))
    if "bench_params" in all_info:  # saved benchmark
        return AppState(
            indices=None,
            worker_state=None,
            session=all_info["result"]["session"],
            default_params=all_info["result"]["session"]["params"],
        )

    else:  # saved web session
        return AppState(
            indices=None,
            worker_state=None,
            session=all_info["session"],
            default_params=all_info["session"]["params"],
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
