from typing import Optional, List, Dict, Callable
from ..seesaw_session import Session, make_session
from ..util import reset_num_cpus
import os
import json
from ..dataset_manager import GlobalDataManager
from .common import *


class WebSession:
    """holds the state for a single user state machine. All actions here are serially run.
    API mirrors the one in WebSeesaw for single user operations
    """

    session_id: str
    session: Optional[Session]
    worker: Optional[Worker]

    def __init__(
        self, root_dir, save_path, session_id, dataset, worker: Worker = None, num_cpus=None
    ):
        if num_cpus is not None:
            reset_num_cpus(num_cpus)

        self.session_id = session_id
        self.root_dir = root_dir
        self.save_path = save_path
        self.worker = worker

        self.gdm = GlobalDataManager(root_dir)
        self.indices = self.gdm.list_indices(dataset)
        self.session = None
        print("web session constructed")

    def _reset_dataset(self, s: SessionParams):
        res = make_session(self.gdm, s)
        self.session = res["session"]

    def next_task(self, body: SessionReq):
        if self.session:  # null the first time
            self.session._log("next_task")
            self.save(body)

        params = self.worker.next_session()
        self._reset_dataset(params)
        return self.getstate()

    def getstate(self):
        return AppState(
            indices=None,
            default_params=None,
            worker_state=self.worker.get_state() if self.worker else None,
            session=self.session.get_state() if self.session else None,
        )

    def reset(self, r: ResetReq):
        if r.config is not None:
            self._reset_dataset(r.config)
        return self.getstate()

    def next(self, body: SessionReq):
        # self.save(body)
        state = body.client_data.session
        if state is not None:  ## refinement code
            self.session.update_state(state)
            self.session.refine()
        self.session.next()
        return self.getstate()

    def text(self, key: str):
        self.session.set_text(key=key)
        self.session.next()
        return self.getstate()

    def save(self, body: SessionReq = None):
        if self.session is not None:
            if body and body.client_data and body.client_data.session:
                self.session.update_state(body.client_data.session)

            self.session._log("save")
            qkey = self.session.params.other_params.get("qkey", None)

            # ensure session id is set correctly in json for easier access at read time
            self.session.params.other_params["session_id"] = self.session_id
            save_time = time.strftime("%Y%m%d-%H%M%S")
            self.session.params.other_params["save_time"] = save_time

            if qkey not in g_queries:
                qkey = "other"

            output_path = f"{self.save_path}/session_{self.session_id}/qkey_{qkey}/saved_{save_time}"
            os.makedirs(output_path, exist_ok=True)
            base = self.getstate().dict()
            json.dump(base, open(f"{output_path}/summary.json", "w"))
            print(f"saved session {output_path}")
            return SaveResp(path="")

    def sleep(self):
        start = time.time()
        time.sleep(10)
        end = time.time()
        return end - start

    def test(self):
        return True
