import json
from seesaw.query_interface import AccessMethod
import numpy as np
import pandas as pd
from .dataset_manager import GlobalDataManager, SeesawDatasetManager

from typing import Optional, List
from pydantic import BaseModel
import os
import time
import pyroaring as pr

def get_image_paths(image_root, path_array, idxs):
    return [ os.path.normpath(f'{image_root}/{path_array[int(i)]}').replace('//', '/') for i in idxs ]

from .search_loop_models import *
from .search_loop_tools import *

import time

from .dataset_tools import *
from .vloop_dataset_loaders import EvDataset, get_class_ev
from .fine_grained_embedding import *
from .search_loop_models import adjust_vec, adjust_vec2
import numpy as np
import sklearn.metrics
import math
from .util import *
from .pairwise_rank_loss import VecState
import pyroaring as pr
from .dataset_manager import VectorIndex
from .query_interface import *
from .multiscale_index import MultiscaleIndex
from .coarse_index import CoarseIndex

from dataclasses import dataclass,field

## used only to make life easier
@dataclass(frozen=True)
class SessionParams:
    interactive : str
    warm_start : str
    batch_size : int
    minibatch_size : int
    learning_rate : float
    max_examples : int
    loss_margin : float
    tqdm_disabled : bool
    granularity : str
    positive_vector_type : str
    num_epochs : int
    n_augment : int
    min_box_size : int = 10
    model_type : int = 'logistic'
    solver_opts : dict = None

from .query_interface import AccessMethod

@dataclass
class LoopState:
    tvec : np.ndarray = None
    tmod : str = None
    latency_profile : list = field(default_factory=list)
    vec_state : VecState = None

class SeesawLoop:
    q : InteractiveQuery
    params : SessionParams
    state : LoopState

    def __init__(self, q : InteractiveQuery, params : SessionParams):
        self.params = params
        self.state = LoopState()
        self.q = q

                #res = {'indices':acc_indices, 'results':acc_results}#, 'gt':gt.values.copy()}

    def next_batch(self):
        """
        gets next batch of image indices based on current vector
        """
        start_lookup = time.time()

        s = self.state
        p = self.params
        lp = {'n_images':None, 'n_posvecs':None, 'n_negvecs':None,
                                    'lookup':None, 'label':None, 'refine':None, }    
        s.latency_profile.append(lp)

        idxbatch, _ = self.q.query_stateful(mode=s.tmode, vector=s.tvec, batch_size=p.batch_size)
        lp['n_images'] = idxbatch.shape[0]
        lp['lookup'] = time.time() - start_lookup
        return idxbatch

    def refine(self):
        """
        update based on vector. box dict will have every index from idx batch, including empty dfs.
        """
        start_refine = time.time()

        p = self.params
        s = self.state
        lp = s.latency_profile[-1]
        
        lp['label'] = start_refine - lp['lookup']

        if p.interactive != 'plain':
            Xt,yt = self.q.getXy()
            lp['n_posvecs'] = (yt == 1).sum()#.shape[0]
            lp['n_negvecs'] = (yt != 1).sum()

            if (yt.shape[0] > 0) and (yt.max() > yt.min()):
                s.tmode = 'dot'
                if p.interactive == 'sklearn':
                    lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
                    lr.fit(Xt, yt)
                    s.tvec = lr.coef_.reshape(1,-1)        
                elif p.interactive == 'pytorch':
                    prob = yt.sum()/yt.shape[0]
                    w = np.clip((1-prob)/prob, .1, 10.)

                    if p.model_type == 'logistic':
                        mod = PTLogisiticRegression(Xt.shape[1], learning_ratep=p.learning_rate, C=0, 
                                                    positive_weight=w)
                        if p.warm_start == 'warm':
                            iv = torch.from_numpy(s.tvec)
                            iv = iv / iv.norm()
                            mod.linear.weight.data = iv.type(mod.linear.weight.dtype)
                        elif p.warm_start == 'default':
                            pass

                        fit_reg(mod=mod, X=Xt.astype('float32'), y=yt.astype('float'), batch_size=p.minibatch_size)
                        s.tvec = mod.linear.weight.detach().numpy().reshape(1,-1)
                    elif p.model_type in ['cosine', 'multirank']:
                        for i in range(p.num_epochs):
                            s.tvec = adjust_vec(s.tvec, Xt, yt, learning_rate=p.learning_rate, 
                                                max_examples=p.max_examples, 
                                                minibatch_size=p.minibatch_size,
                                                loss_margin=p.loss_margin)
                    elif p.model_type in ['multirank2']:
                        npairs = yt.sum() * (1-yt).sum()
                        max_iters = math.ceil(min(npairs, p.max_examples)//p.minibatch_size) * p.num_epochs
                        print('max iters this round would have been', max_iters)
                        #print(s.vec_state.)

                        # vecs * niters = number of vector seen.
                        # n vec seen <= 10000
                        # niters <= 10000/vecs
                        max_vec_seen = 10000
                        n_iters = math.ceil(max_vec_seen/Xt.shape[0])
                        n_steps = np.clip(n_iters, 20, 200)

                        # print(f'steps for this iteration {n_steps}. num vecs: {Xt.shape[0]} ')
                        # want iters * vecs to be const..
                        # eg. dota. 1000*100*30

                        for _ in range(n_steps):
                            loss = s.vec_state.update(Xt, yt)
                            if loss == 0: # gradient is 0 when loss is 0.
                                print('loss is 0, breaking early')
                                break

                        s.tvec = s.vec_state.get_vec()
                    elif p.model_type == 'solver':
                        s.tvec = adjust_vec2(s.tvec, Xt, yt, **p.solver_opts)
                    else:
                        assert False, 'model type'
                else:
                    assert False
            else:
                # print('missing positives or negatives to do any training', yt.shape, yt.max(), yt.min())
                pass

            lp['refine'] = time.time() - start_refine


class Imdata(BaseModel):
    url : str
    dbidx : int
    boxes : Optional[List[Box]] # None means not labelled (neutral). [] means positively no boxes.
    refboxes : Optional[List[Box]]
    marked_accepted : bool

class SessionState(BaseModel):
    gdata : List[List[Imdata]]
    timing : List[float]
    reference_categories : List[str]

class Session:
    current_dataset : str
    current_index : str
    loop : SeesawLoop
    acc_indices : list
    total_results : int
    timing : list
    accepted : pr.BitMap
    q : InteractiveQuery
    index : AccessMethod

    def __init__(self, gdm : GlobalDataManager, dataset : SeesawDatasetManager, hdb : AccessMethod, params : SessionParams):
        self.gdm = gdm
        self.dataset = dataset
        self.acc_indices = []
        self.accepted = pr.BitMap()
        self.params = params
        self.init_q = None
        self.timing = []
        self.index = hdb 
        self.q = hdb.new_query()

        self.loop = SeesawLoop(self.q, params=self.params)

    def next(self):
        start = time.time()
        idxbatch = self.loop.next_batch()

        delta = time.time() - start

        self.acc_indices.append(idxbatch)
        self.timing.append(delta)

        return idxbatch

    def set_text(self, key):        
        self.init_q = key
        p = self.loop.params
        s = self.loop.state

        if key == 'nolang':
            s.tvec = None
            s.tmode = 'random'
        else:
            init_vec = self.index.string2vec(string=key)
            s.tvec = init_vec
            s.tmode = 'dot'
            if p.model_type == 'multirank2':
                s.vec_state = VecState(init_vec, margin=p.loss_margin, opt_class=torch.optim.SGD, 
                opt_params={'lr':p.learning_rate})

    def update_state(self, state: SessionState):
        self._update_labeldb(state.gdata)

    def refine(self):
        self.loop.refine()

    def get_state(self) -> SessionState:
        gdata = []
        for indices in self.acc_indices:
            imdata = self.get_panel_data(idxbatch=indices)
            gdata.append(imdata)
        
        dat = {'gdata':gdata}
        dat['timing']  = self.timing
        # if self.ev.query_ground_truth is not None:
        #     dat['reference_categories'] = self.ev.query_ground_truth.columns.values.tolist()
        # else:
        dat['reference_categories'] = []
        return SessionState(**dat)

    def get_panel_data(self, *, idxbatch):
        reslabs = []
        urls = get_image_paths(self.dataset.image_root, self.dataset.paths, idxbatch)

        for (url, dbidx) in zip(urls, idxbatch):
            dbidx = int(dbidx)
            boxes = self.q.label_db.get(dbidx, format='box') # None means no annotations yet (undef), empty means no boxes.
            elt = Imdata(url=url, dbidx=dbidx, boxes=boxes, refboxes=None, marked_accepted=dbidx in self.accepted)
            reslabs.append(elt)
        return reslabs

    def _update_labeldb(self, gdata):
        for ldata in gdata:
            for imdata in ldata:
                if imdata.marked_accepted:
                    self.accepted.add(imdata.dbidx)
                self.q.label_db.put(imdata.dbidx, imdata.boxes)

    def save_state(self, path):
        os.makedirs(path, exist_ok=True)
        json.dump(self.params.__dict__,open(f'{path}/loop_params.json', 'w'))
        st = self.get_state()
        json.dump(st.dict(), open(f'{path}/session_state.json', 'w'))