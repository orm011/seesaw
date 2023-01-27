from .loops.loop_base import *
from .loops.registry import build_loop_from_params
from .dataset import BaseDataset
from .indices.interface import AccessMethod
from .labeldb import LabelDB
from .basic_types import SessionState, SessionParams, ActivationData, Box, Imdata, is_image_accepted

import time
import pyroaring as pr

class Session:
    current_dataset: str
    current_index: str
    loop: LoopBase
    acc_indices: list
    image_timing: dict
    acc_activations: list
    total_results: int
    timing: list
    seen: pr.BitMap
    accepted: pr.BitMap
    q: InteractiveQuery
    index: AccessMethod

    def __init__(
        self,
        gdm: GlobalDataManager,
        dataset: BaseDataset,
        hdb: AccessMethod,
        params: SessionParams,
    ):
        self.gdm = gdm
        self.dataset = dataset
        self.acc_indices = []
        self.acc_activations = []
        self.seen = pr.BitMap([])
        self.accepted = pr.BitMap([])
        self.params = params
        self.init_q = None
        self.timing = []
        self.image_timing = {}
        self.index = hdb
        self.q = hdb.new_query()
        self.label_db = LabelDB() #prefilled one. not the main one used in q.
        if self.params.annotation_category != None:
            box_data, _ = self.dataset.load_ground_truth()
            mask = box_data.category == self.params.annotation_category
            if mask.sum() == 0:
                print(f'warning, no entries found for category {self.params.annotation_category}. if you expect some check for typos')
            df = box_data[box_data.category == self.params.annotation_category]
            self.label_db.fill(df)

        self.loop = build_loop_from_params(self.gdm, self.q, params=self.params)
        self.action_log = []
        self._log("init")

    def get_totals(self):
        return {"seen": len(self.seen), "accepted": len(self.accepted)}

    def _log(self, message: str):
        self.action_log.append(
            {
                "logger": "server",
                "time": time.time(),
                "message": message,
                "seen": len(self.seen),
                "accepted": len(self.accepted),
            }
        )

    def next(self):
        self._log("next.start")

        start = time.time()
        r = self.loop.next_batch()

        delta = time.time() - start

        self.acc_indices.append(r["dbidxs"])
        self.acc_activations.append(r["activations"])
        self.timing.append(delta)
        self._log("next.end")
        return r["dbidxs"]

    def set_text(self, key):
        self._log("set_text")
        self.init_q = key
        s = self.loop.state
        s.curr_str = key

        vec = self.index.string2vec(string=key)
        self.loop.set_text_vec(vec)

    def update_state(self, state: SessionState):
        self._update_labeldb(state)
        self._log(
            "update_state.end"
        )  # log this after updating so that it includes all new information

    def refine(self):
        self._log("refine.start")
        self.loop.refine()
        self._log("refine.end")

    def get_state(self) -> SessionState:
        gdata = []
        for i, (indices, accs) in enumerate(zip(self.acc_indices, self.acc_activations)):
            prefill = (self.params.annotation_category is not None) and (i == len(self.acc_indices) - 1)
             # prefill last batch if annotation category is on (assumes last batch has no user annotations yet..)
            imdata = self.get_panel_data(idxbatch=indices, activation_batch=accs, prefill=prefill)
            gdata.append(imdata)
        dat = {}
        dat["action_log"] = self.action_log
        dat["gdata"] = gdata
        dat["timing"] = self.timing
        dat["reference_categories"] = []
        dat["params"] = self.params
        dat["query_string"] = self.loop.state.curr_str
        return SessionState(**dat)

    def get_panel_data(self, *, idxbatch, activation_batch=None, prefill=False):
        reslabs = []
        #urls = get_image_paths(self.dataset.image_root, self.dataset.paths, idxbatch)
        urls = self.dataset.get_urls(idxbatch)

        for i, (url, dbidx) in enumerate(zip(urls, idxbatch)):
            dbidx = int(dbidx)

            if prefill:
                boxes = self.label_db.get(dbidx, format="box")
                # None means no boxes.
            else:
                boxes = self.q.label_db.get(
                    dbidx, format="box"
                )  # None means no annotations yet (undef), empty means no boxes.

            if activation_batch is None:
                activations = None
            else:
                activations = []
                for row in activation_batch[i].to_dict(orient="records"):
                    score = row["score"]
                    del row["score"]
                    activations.append(ActivationData(box=Box(**row), score=score))

            elt = Imdata(
                url=url,
                dbidx=dbidx,
                boxes=boxes,
                activations=activations,
                timing=self.image_timing.get(dbidx, []),
            )
            reslabs.append(elt)
        return reslabs

    def _update_labeldb(self, state: SessionState):
        ## clear bitmap and reconstruct bc user may have erased previously accepted images
        self.action_log = state.action_log  # just replace the log
        gdata = state.gdata
        self.accepted.clear()
        self.seen.clear()
        for ldata in gdata:
            for imdata in ldata:
                self.image_timing[imdata.dbidx] = imdata.timing
                self.seen.add(imdata.dbidx)
                if is_image_accepted(imdata):
                    self.accepted.add(imdata.dbidx)
                self.q.label_db.put(imdata.dbidx, imdata.boxes)

def get_labeled_subset_dbdidxs(qgt, c_name):
    labeled = ~qgt[c_name].isna()
    return qgt[labeled].index.values

def make_session(gdm: GlobalDataManager, p: SessionParams):
    ds = gdm.get_dataset(p.index_spec.d_name)
    if p.index_spec.c_name is not None:
        print('subsetting...')
        ds = ds.load_subset(p.index_spec.c_name)
        print('done subsetting')

    print('loading index')
    idx = ds.load_index(p.index_spec.i_name, options=p.index_options)
    print('done loading index')
    
    print("about to construct session...")
    session = Session(gdm, ds, idx, p)
    print("session constructed...")
    return {
        "session": session,
        "dataset": ds,
    }
