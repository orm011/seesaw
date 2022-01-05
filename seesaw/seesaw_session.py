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
class LoopParams:
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


def make_acccess_method(ev, p : LoopParams):
    if p.granularity == 'multi':
        hdb = MultiscaleIndex(images=ev.image_dataset, embedding=ev.embedding, 
            vectors=ev.fine_grained_embedding, vector_meta=ev.fine_grained_meta, vec_index=ev.vec_index)
    elif p.granularity == 'coarse':
        hdb = CoarseIndex(images=ev.image_dataset, embedding=ev.embedding,vectors=ev.embedded_dataset)
    else:
        assert False

    return hdb

class SeesawLoop:
    bfq : InteractiveQuery
    params : LoopParams
    state : LoopState

    def __init__(self, hdb : AccessMethod, params : LoopParams):
        self.params = params
        self.state = LoopState()

        p = self.params
        s = self.state

        self.hdb = hdb
        self.bfq = hdb.new_query()

    def set_vec(self, qstr : str):
        p = self.params
        s = self.state

        if qstr == 'nolang':
            s.tvec = None
            s.tmode = 'random'
        else:
            init_vec = self.hdb.string2vec(string=qstr)
            s.tvec = init_vec
            s.tmode = 'dot'
            if p.model_type == 'multirank2':
                s.vec_state = VecState(init_vec, margin=p.loss_margin, opt_class=torch.optim.SGD, 
                opt_params={'lr':p.learning_rate})
        
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

        idxbatch, _ = self.bfq.query_stateful(mode=s.tmode, vector=s.tvec, batch_size=p.batch_size)
        lp['n_images'] = idxbatch.shape[0]
        lp['lookup'] = time.time() - start_lookup
        return idxbatch

    def refine(self, idxbatch : np.array, box_dict : dict):
        """
        update based on vector. box dict will have every index from idx batch, including empty dfs.
        """
        assert idxbatch.shape[0] == len(box_dict)

        start_refine = time.time()

        p = self.params
        s = self.state
        lp = s.latency_profile[-1]
        
        lp['label'] = start_refine - lp['lookup']

        if p.interactive != 'plain':
            Xt,yt = self.bfq.getXy(idxbatch, box_dict)
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