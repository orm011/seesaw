from typing import Optional, List, Dict, Callable
from fastapi.applications import FastAPI
from pydantic import BaseModel
from fastapi import Body, FastAPI, HTTPException

from fastapi import HTTPException

from ..basic_types import Box, SessionState, SessionParams, IndexSpec

import time

from ..configs import (
    make_session_params
)


class TaskParams(BaseModel):
    task_index: int
    qkey: str
    mode: str
    qstr: str
    dataset: str


class WorkerState(BaseModel):
    task_list: List[TaskParams]
    current_task_index: int


class AppState(BaseModel):  # Using this as a response for every state transition.
    indices: Optional[List[IndexSpec]]
    worker_state: Optional[WorkerState]
    default_params: Optional[SessionParams]
    session: Optional[SessionState]  # sometimes there is no active session
    save_path : Optional[str]


class SearchDesc(BaseModel):
    dataset: str
    qstr: str
    description: str = ""
    negative_description: Optional[str]


class NotificationState(BaseModel):
    urls: List[str]
    neg_urls: List[str]
    description: SearchDesc


class SessionReq(BaseModel):
    client_data: AppState


class ResetReq(BaseModel):
    config: Optional[SessionParams]


class SessionInfoReq(BaseModel):
    path: str


class SaveResp(BaseModel):
    path: str


class EndSession(BaseModel):
    token: Optional[str]


def session_params(mode, dataset, index, **kwargs):
    # assert mode in _session_modes.keys()
    base = make_session_params(mode, dataset, index)
    base.other_params = {"mode": mode, "dataset": dataset, **kwargs}
    return base


class Worker:
    session_id: str
    task_list: List[TaskParams]
    current_task: int

    def __init__(self, session_id, task_list):
        self.session_id = session_id
        self.task_list = task_list
        self.current_task = -1

    def get_state(self) -> WorkerState:
        return WorkerState(
            task_list=self.task_list, current_task_index=self.current_task
        )

    def next_session(self):
        self.current_task += 1
        task = self.task_list[self.current_task]
        new_params = session_params(**task.dict())
        return new_params


g_queries = {
    "pc": SearchDesc(
        dataset="bdd",
        qstr="police cars",
        description="""Police vehicles that have lights and some marking related to police. """,
        negative_description="""Sometimes private security vehicles or ambulances look like police cars but should not be included""",
    ),
    "dg": SearchDesc(dataset="bdd", qstr="dogs"),
    "cd": SearchDesc(
        dataset="bdd",
        qstr="car with open doors",
        description="""Any vehicles with any open doors, including open trunks in cars, and rolled-up doors in trucks and trailers.""",
        negative_description="""We dont count rolled down windows as open doors""",
    ),
    "wch": SearchDesc(
        dataset="bdd",
        qstr="wheelchairs",
        description="""We include wheelchair alternatives such as electric scooters for the mobility impaired. """,
        negative_description="""We do not include wheelchair signs or baby strollers""",
    ),
    "mln": SearchDesc(
        dataset="coco",
        qstr="cantaloupe or honeydew melon",
        description="""We inclulde both cantaloupe (orange melon) and honeydew (green melon), whole melons and melon pieces. """,
        negative_description="""We dont include any other types of melon, including watermelons, papaya or pumpkins, which can look similar. 
                If you cannot tell whether a fruit piece is really from melon don't sweat it and leave it out.""",
    ),
    "spn": SearchDesc(
        dataset="coco",
        qstr="spoons or teaspoons",
        description="""We include spoons or teaspons of any material for eating. """,
        negative_description="""We dont include the large cooking or serving spoons, ladles for soup, or measuring spoons.""",
    ),
    "dst": SearchDesc(
        dataset="objectnet",
        qstr="dustpans",
        description="""We include dustpans on their own or together with other tools, like brooms, from any angle.""",
        negative_description="""We dont include brooms alone""",
    ),
    "gg": SearchDesc(
        dataset="objectnet",
        qstr="egg cartons",
        description="""These are often made of cardboard or styrofoam. We include them viewed from any angle.""",
        negative_description="""We dont include the permanent egg containers that come in the fridge""",
    ),
}
