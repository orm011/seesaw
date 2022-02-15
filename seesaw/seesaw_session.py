import json
from seesaw.query_interface import AccessMethod
import numpy as np
import pandas as pd
from .dataset_manager import GlobalDataManager, SeesawDatasetManager

import os
import time
import numpy as np
import sklearn.metrics
import math
import pyroaring as pr
from dataclasses import dataclass,field

def get_image_paths(image_root, path_array, idxs):
    return [ os.path.normpath(f'{image_root}/{path_array[int(i)]}').replace('//', '/') for i in idxs ]

from .basic_types import *
from .search_loop_models import *
from .search_loop_tools import *
from .dataset_tools import *
from .fine_grained_embedding import *
from .search_loop_models import adjust_vec, adjust_vec2
from .util import *
from .pairwise_rank_loss import VecState
from .query_interface import *
from .textual_feedback_box import OnlineModel,  join_vecs2annotations

@dataclass
class LoopState:
    curr_str : str = None
    tvec : np.ndarray = None
    tmod : str = None
    latency_profile : list = field(default_factory=list)
    vec_state : VecState = None
    model : OnlineModel = None

class SeesawLoop:
    q : InteractiveQuery
    params : SessionParams
    state : LoopState

    def __init__(self, gdm : GlobalDataManager, q : InteractiveQuery, params : SessionParams):
        self.params = params
        self.state = LoopState()
        self.q = q

        if self.params.interactive in ['textual']:
          param_dict = gdm.global_cache.read_state_dict('/home/gridsan/groups/fastai/omoll/seesaw_root/models/clip/ViT-B-32.pt', jit=True)
          self.state.model = OnlineModel(param_dict, self.params.method_config)

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

        if p.interactive == 'textual':
          if p.method_config['mode'] == 'finetune':
            vec = s.model.encode_string(s.curr_str)
            rescore_m = lambda vecs : vecs @ vec.reshape(-1, 1)
          elif p.method_config['mode'] == 'linear':
            if len(s.model.linear_scorer.scorers) == 0: ## first time
              vec = s.model.encode_string(s.curr_str)
              s.model.linear_scorer.add_scorer(s.curr_str,torch.from_numpy(vec.reshape(-1)))
            rescore_m = self.state.model.score_vecs
            vec = self.state.model.get_lookup_vec(s.curr_str)
        else:
          vec = s.tvec
          rescore_m = lambda vecs : vecs @ vec.reshape(-1, 1)

        b =  self.q.query_stateful(mode=s.tmode, vector=vec, batch_size=p.batch_size, shortlist_size=p.shortlist_size, 
                    agg_method=p.agg_method,
                    rescore_method=rescore_m)

        idxbatch = b['dbidxs']
        lp['n_images'] = idxbatch.shape[0]
        lp['lookup'] = time.time() - start_lookup
        return b

    def compute_image_activatins(self, dbidx, annotations):
        pass

    def refine(self):
        """
        update based on vector. box dict will have every index from idx batch, including empty dfs.
        """
        start_refine = time.time()

        p = self.params
        s = self.state
        lp = s.latency_profile[-1]
        
        lp['label'] = start_refine - lp['lookup']
        
        if p.interactive == 'plain':
          return
        elif p.interactive in ['textual']:
          # get vectors and string corresponding to current annotations
          # run the updater.
          print('textual update')

          if 'image_vector_strategy' not in p.dict() or \
              p.image_vector_strategy == None or p.image_vector_strategy == 'matched':
            vecs = []
            strs = []
            acc = []

            for dbidx in self.q.label_db.get_seen():
              annot = self.q.label_db.get(dbidx, format='box')
              assert annot is not None
              if len(annot) == 0:
                continue

              dfvec, dfbox = join_vecs2annotations(self.q.index, dbidx, annot)
              # best_box_iou, best_box_idx
              
              ## vectors with overlap
              df = dfbox # use boxes as guide for now
              mask_boxes = df.best_box_iou > p.method_config['vector_box_min_iou']
              print(f'using {mask_boxes.sum()} out of {mask_boxes.shape[0]} given masks')
              df = df[mask_boxes]
              if df.shape[0] > 0:
                vecs.append(df.vectors.values)
                strs.append(df.descriptions.values)
                acc.append(df.marked_accepted.values)

            if len(vecs) == 0:
              print('no annotations for update... skipping')
              return

            all_vecs = np.concatenate(vecs)
            all_strs = np.concatenate(strs)
            marked_accepted = np.concatenate(acc)
          elif p.image_vector_strategy == 'computed':
              vecs = []
              strs = []
              acc = []
              #annot = self.q.label_db.get(dbidx, format='box')
              for dbidx in self.q.label_db.get_seen():
                annot = self.q.label_db.get(dbidx, format='box')
                if len(annot) == 0:
                  continue

                vecs.append(self.compute_image_activations(dbidx, annot))
                strs.append()

              pass
          else:
            assert False, 'unknown image vec strategy'

          losses = s.model.update(all_vecs, marked_accepted, all_strs, s.curr_str)
          print('done with update', losses)
        else:
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

                    cfg = p.method_config

                    if cfg['model_type'] == 'logistic':
                        mod = PTLogisiticRegression(Xt.shape[1], learning_ratep=p.learning_rate, C=0, 
                                                    positive_weight=w)
                        if cfg['warm_start'] == 'warm':
                            iv = torch.from_numpy(s.tvec)
                            iv = iv / iv.norm()
                            mod.linear.weight.data = iv.type(mod.linear.weight.dtype)
                        elif cfg['warm_start'] == 'default':
                            pass

                        fit_reg(mod=mod, X=Xt.astype('float32'), y=yt.astype('float'), batch_size=p.minibatch_size)
                        s.tvec = mod.linear.weight.detach().numpy().reshape(1,-1)
                    elif cfg['model_type'] in ['cosine', 'multirank']:
                        for i in range(cfg['num_epochs']):
                            s.tvec = adjust_vec(s.tvec, Xt, yt, learning_rate=cfg['learning_rate'], 
                                                max_examples=cfg['max_examples'], 
                                                minibatch_size=cfg['minibatch_size'],
                                                loss_margin=cfg['loss_margin'])
                    elif cfg['model_type'] in ['multirank2']:
                        npairs = yt.sum() * (1-yt).sum()
                        max_iters = math.ceil(min(npairs, cfg['max_examples'])//cfg['minibatch_size']) * cfg['num_epochs']
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
                    elif cfg['model_type'] == 'solver':
                        s.tvec = adjust_vec2(s.tvec, Xt, yt, **p.solver_opts)
                    else:
                        assert False, 'model type'
                else:
                    assert False
            else:
                # print('missing positives or negatives to do any training', yt.shape, yt.max(), yt.min())
                pass


class Session:
    current_dataset : str
    current_index : str
    loop : SeesawLoop
    acc_indices : list
    acc_activations : list
    total_results : int
    timing : list
    accepted : pr.BitMap
    q : InteractiveQuery
    index : AccessMethod

    def __init__(self, gdm : GlobalDataManager, dataset : SeesawDatasetManager, hdb : AccessMethod, params : SessionParams):
        self.gdm = gdm
        self.dataset = dataset
        self.acc_indices = []
        self.acc_activations = []
        self.accepted = pr.BitMap()
        self.params = params
        self.init_q = None
        self.timing = []
        self.index = hdb 
        self.q = hdb.new_query()

        self.loop = SeesawLoop(self.gdm, self.q, params=self.params)

    def next(self):
        start = time.time()
        r = self.loop.next_batch()

        delta = time.time() - start

        self.acc_indices.append(r['dbidxs'])
        self.acc_activations.append(r['activations'])
        self.timing.append(delta)

        return r['dbidxs']

    def set_text(self, key):        
        self.init_q = key
        p = self.loop.params
        s = self.loop.state
        s.curr_str = key
        s.tvec = self.index.string2vec(string=key)
        s.tmode = 'dot'
        if p.method_config.get('model_type',None) == 'multirank2':
            s.vec_state = VecState(s.tvec, margin=p.loss_margin, opt_class=torch.optim.SGD, 
            opt_params={'lr':p.learning_rate})

    def update_state(self, state: SessionState):
        self._update_labeldb(state.gdata)

    def refine(self):
        self.loop.refine()

    def get_state(self) -> SessionState:
        gdata = []
        for indices,accs in zip(self.acc_indices, self.acc_activations):
            imdata = self.get_panel_data(idxbatch=indices, activation_batch=accs)
            gdata.append(imdata)
        
        dat = {}
        dat['gdata'] = gdata
        dat['timing']  = self.timing
        dat['reference_categories'] = []
        dat['params'] = self.params
        dat['query_string'] = self.loop.state.curr_str
        return SessionState(**dat)

    def get_panel_data(self, *, idxbatch, activation_batch=None):
        reslabs = []
        urls = get_image_paths(self.dataset.image_root, self.dataset.paths, idxbatch)

        for i, (url, dbidx) in enumerate(zip(urls, idxbatch)):
            dbidx = int(dbidx)
            boxes = self.q.label_db.get(dbidx, format='box') # None means no annotations yet (undef), empty means no boxes.
            if activation_batch is None:
              activations = None
            else:
              activations = []
              for row in activation_batch[i].to_dict(orient='records'):
                score = row['score']
                del row['score']
                activations.append(ActivationData(box=Box(**row), score=score))

            elt = Imdata(url=url, dbidx=dbidx, boxes=boxes, activations=activations)
            reslabs.append(elt)
        return reslabs

    def _update_labeldb(self, gdata):
        for ldata in gdata:
            for imdata in ldata:
                if is_image_accepted(imdata):
                    self.accepted.add(imdata.dbidx)
                self.q.label_db.put(imdata.dbidx, imdata.boxes)

    def get_data(self):
        #os.makedirs(path, exist_ok=True)
        #json.dump(self.params.__dict__,open(f'{path}/loop_params.json', 'w'))
        st = self.get_state()
        return dict(session_params=self.params.dict(), session_state=st.dict())


def prep_data(ds, p : SessionParams):
    box_data, qgt = ds.load_ground_truth()
    catgt = qgt[p.index_spec.c_name]    

    positive_box_data = box_data[box_data.category == p.index_spec.c_name]
    present = pr.FrozenBitMap(catgt[~catgt.isna()].index.values)
    positive = pr.FrozenBitMap(positive_box_data.dbidx.values)

    assert positive.intersection(present) == positive    
    return box_data, present, positive

def make_session(gdm, p : SessionParams):
    ds = gdm.get_dataset(p.index_spec.d_name)
    hdb = gdm.load_index(p.index_spec.d_name, p.index_spec.i_name)
    print('index loaded')
    box_data = None
    subset = None
    positive = None

    if p.index_spec.c_name is not None:
      print('prepping  data....')
      box_data, subset, positive  = prep_data(ds, p)
      assert len(positive) != 0
      hdb = hdb.subset(subset)
    
    print('about to construct session...')
    session = Session(gdm, ds, hdb, p)
    print('session constructed...')
    return {'session':session, 'box_data':box_data, 'subset':subset, 'positive':positive}