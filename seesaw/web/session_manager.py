import ray
from typing import Dict
from ray.actor import ActorHandle
import random, string
import ray

from .common import *
from .web_session_actor import WebSession


def generate_task_list(mode):
    tasks = []
    # qs = random.shuffle(g_queries.items())
    # for q in :
    for i, k in enumerate(g_queries.keys()):
        sdesc = g_queries[k]
        task = TaskParams(
            mode=mode, qkey=k, qstr=sdesc.qstr, dataset=sdesc.dataset, task_index=i
        )
        tasks.append(task)

    return tasks


def generate_id():
    return "".join(
        random.choice(string.ascii_letters + "0123456789") for _ in range(32)
    )


WebSessionActor = ray.remote(WebSession)


class SessionManager:
    sessions: Dict[str, ActorHandle]

    def __init__(self, root_dir, save_path, num_cpus_per_session):
        self.root_dir = root_dir
        self.save_path = save_path
        self.num_cpus = num_cpus_per_session
        self.sessions = {}

    def ready(self):
        return True

    def _new_session(self, task_list):
        session_id = generate_id()
        worker = Worker(session_id=session_id, task_list=task_list)
        self.sessions[session_id] = WebSessionActor.options(
            name=f"web_session#{session_id}", num_cpus=self.num_cpus
        ).remote(
            self.root_dir, self.save_path, session_id, worker, num_cpus=self.num_cpus
        )
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
        print(f"ending session {session_id}")
        ray.kill(sess)

    def get_session(self, session_id):
        return self.sessions.get(session_id)


SessionManagerActor = ray.remote(SessionManager)


def get_manager():
    return ray.get_actor("session_manager")
