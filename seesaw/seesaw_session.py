import json

from torch import index_fill
from seesaw.query_interface import AccessMethod
import numpy as np
import pandas as pd
from .dataset_manager import GlobalDataManager, SeesawDataset

import os
import time
import numpy as np
import sklearn.metrics
import math
import pyroaring as pr
from dataclasses import dataclass, field


def get_image_paths(image_root, path_array, idxs):
    return [
        os.path.normpath(f"{image_root}/{path_array[int(i)]}").replace("//", "/")
        for i in idxs
    ]


from .basic_types import *
from .search_loop_models import *
from .search_loop_tools import *
from .dataset_tools import *
from .fine_grained_embedding import *
from .search_loop_models import adjust_vec, adjust_vec2
from .util import *
from .pairwise_rank_loss import VecState
from .query_interface import *
from .research.knn_methods import KNNGraph, LabelPropagationRanker2, SimpleKNNRanker
from .indices.multiscale.multiscale_index import _get_top_dbidxs, score_frame2

@dataclass
class LoopState:
    curr_str: str = None
    tvec: np.ndarray = None
    vec_state: VecState = None
    # model: OnlineModel = None
    knn_model : SimpleKNNRanker = None

class LoopBase:
    q: InteractiveQuery
    params: SessionParams
    state: LoopState

    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        self.gdm = gdm
        self.params = params
        self.state = LoopState()
        self.q = q
        self.index = self.q.index

    def set_text_vec(self, tvec):
        pass

    @staticmethod
    def from_params(gdm, q, params):

        if params.interactive in ['knn_greedy', 'knn_prop', 'knn_prop2', 'linear_prop']:
            return KnnBased.from_params(gdm, q, params)
        elif params.interactive in ['textual']:
            cls = TextualLoop
        elif params.interactive in ['plain']:
            cls = PointBased
        elif params.interactive in ['pytorch']:
            cls = SeesawLoop
        elif params.interactive == 'log_reg2':
            cls = LogReg2
        elif params.interactive == 'pseudo_lr':
            cls = PseudoLabelLR
        else:
            assert False, f'unknown {params.interactive=}'

        return cls(gdm, q, params)

    def next_batch(self):
        pass

    def refine(self):
        pass

class PointBased(LoopBase):
    def __init__(self, gdm, q, params):
        super().__init__(gdm, q, params)
        self.curr_vec = None

    def set_text_vec(self, vec):
        self.state.tvec = vec
        self.curr_vec = vec

    def refine(self):
        pass # modify in subclasses

    def next_batch(self):
        s = self.state
        p = self.params

        vec = self.curr_vec
        rescore_m = lambda vecs: vecs @ vec.reshape(-1, 1)

        b = self.q.query_stateful(
            vector=vec,
            batch_size=p.batch_size,
            shortlist_size=p.shortlist_size,
            agg_method=p.agg_method,
            aug_larger=p.aug_larger,
            rescore_method=rescore_m,
        )

        return b
        
class TextualLoop(LoopBase):
    def __init__(self, gdm, q, params):
        super().__init__(gdm, q, params)
        s = self.state
        p = self.params
        
        if self.params.interactive in ["textual"]:
            param_dict = gdm.global_cache.read_state_dict(
                "/home/gridsan/groups/fastai/omoll/seesaw_root/models/clip/ViT-B-32.pt",
                jit=True,
            )
            # s.model = OnlineModel(param_dict, p.method_config)

    def set_text_vec(self, tvec):
        pass

    def refine(self):
        p = self.params
        s = self.state

        if p.method_config["mode"] == "finetune":
            vec = s.model.encode_string(s.curr_str)
            rescore_m = lambda vecs: vecs @ vec.reshape(-1, 1)
        elif p.method_config["mode"] == "linear":
            if len(s.model.linear_scorer.scorers) == 0:  ## first time
                vec = s.model.encode_string(s.curr_str)
                s.model.linear_scorer.add_scorer(
                    s.curr_str, torch.from_numpy(vec.reshape(-1))
                )
            rescore_m = self.state.model.score_vecs
            vec = self.state.model.get_lookup_vec(s.curr_str)

        b = self.q.query_stateful(
            vector=vec,
            batch_size=p.batch_size,
            shortlist_size=p.shortlist_size,
            agg_method=p.agg_method,
            aug_larger=p.aug_larger,
            rescore_method=rescore_m,
        )

        return b

    def next_batch(self):
        p = self.params
        s = self.state
        if (
            "image_vector_strategy" not in p.dict()
            or p.image_vector_strategy == None
            or p.image_vector_strategy == "matched"
        ):
            vecs = []
            strs = []
            acc = []

            for dbidx in self.q.label_db.get_seen():
                annot = self.q.label_db.get(dbidx, format="box")
                assert annot is not None
                if len(annot) == 0:
                    continue

                dfvec, dfbox = join_vecs2annotations(self.q.index, dbidx, annot)
                # best_box_iou, best_box_idx

                ## vectors with overlap
                df = dfbox  # use boxes as guide for now
                mask_boxes = df.best_box_iou > p.method_config["vector_box_min_iou"]
                df = df[mask_boxes]
                if df.shape[0] > 0:
                    vecs.append(df.vectors.values)
                    strs.append(df.descriptions.values)
                    acc.append(df.marked_accepted.values)

            if len(vecs) == 0:
                print("no annotations for update... skipping")
                return

            all_vecs = np.concatenate(vecs)
            all_strs = np.concatenate(strs)
            marked_accepted = np.concatenate(acc)
        elif p.image_vector_strategy == "computed":
            vecs = []
            strs = []
            acc = []
            # annot = self.q.label_db.get(dbidx, format='box')
            for dbidx in self.q.label_db.get_seen():
                annot = self.q.label_db.get(dbidx, format="box")
                if len(annot) == 0:
                    continue

                vecs.append(self.compute_image_activations(dbidx, annot))
                strs.append()

            pass
        else:
            assert False, "unknown image vec strategy"

        losses = s.model.update(all_vecs, marked_accepted, all_strs, s.curr_str)
        print("done with update", losses)


from .logistic_regression import LogisticRegressionPT
class LogReg2(PointBased):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        super().__init__(gdm, q, params)

    # def set_text_vec(self) # let super do this
    def refine(self):
        Xt, yt = self.q.getXy()
        model = LogisticRegressionPT(regularizer_vector=self.state.tvec, **self.params.interactive_options)
        model.fit(Xt, yt.reshape(-1,1))
        self.curr_vec = model.get_coeff()

class SeesawLoop(PointBased):
    def __init__(
        self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams
    ):
        super().__init__(gdm, q, params)
        p = self.params
        assert p.interactive == 'pytorch'

    def set_text_vec(self, tvec):
        super().set_text_vec(tvec)

        s = self.state
        p = self.params

        if self.params.method_config.get("model_type", None) == "multirank2":
            self.state.vec_state = VecState(
                tvec,
                margin=p.loss_margin,
                opt_class=torch.optim.SGD,
                opt_params={"lr": p.learning_rate},
            )

    def refine(self):
        """
        update based on vector. box dict will have every index from idx batch, including empty dfs.
        """
        s = self.state
        p = self.params

        Xt, yt = self.q.getXy()

        if (yt.shape[0] == 0) or (yt.max() == yt.min()):
            pass # nothing to do yet.

        if p.interactive == "sklearn":
            lr = sklearn.linear_model.LogisticRegression(
                class_weight="balanced"
            )
            lr.fit(Xt, yt)
            s.tvec = lr.coef_.reshape(1, -1)
        elif p.interactive == "pytorch":
            prob = yt.sum() / yt.shape[0]
            w = np.clip((1 - prob) / prob, 0.1, 10.0)

            cfg = p.method_config

            if cfg["model_type"] == "logistic":
                mod = PTLogisiticRegression(
                    Xt.shape[1],
                    learning_ratep=p.learning_rate,
                    C=0,
                    positive_weight=w,
                )
                if cfg["warm_start"] == "warm":
                    iv = torch.from_numpy(s.tvec)
                    iv = iv / iv.norm()
                    mod.linear.weight.data = iv.type(mod.linear.weight.dtype)
                elif cfg["warm_start"] == "default":
                    pass

                fit_reg(
                    mod=mod,
                    X=Xt.astype("float32"),
                    y=yt.astype("float"),
                    batch_size=p.minibatch_size,
                )
                s.tvec = mod.linear.weight.detach().numpy().reshape(1, -1)
            elif cfg["model_type"] in ["cosine", "multirank"]:
                for i in range(cfg["num_epochs"]):
                    s.tvec = adjust_vec(
                        s.tvec,
                        Xt,
                        yt,
                        learning_rate=cfg["learning_rate"],
                        max_examples=cfg["max_examples"],
                        minibatch_size=cfg["minibatch_size"],
                        loss_margin=cfg["loss_margin"],
                    )
            elif cfg["model_type"] in ["multirank2"]:
                npairs = yt.sum() * (1 - yt).sum()
                max_iters = (
                    math.ceil(
                        min(npairs, cfg["max_examples"])
                        // cfg["minibatch_size"]
                    )
                    * cfg["num_epochs"]
                )
                print("max iters this round would have been", max_iters)
                # print(s.vec_state.)

                # vecs * niters = number of vector seen.
                # n vec seen <= 10000
                # niters <= 10000/vecs
                max_vec_seen = 10000
                n_iters = math.ceil(max_vec_seen / Xt.shape[0])
                n_steps = np.clip(n_iters, 20, 200)

                # print(f'steps for this iteration {n_steps}. num vecs: {Xt.shape[0]} ')
                # want iters * vecs to be const..
                # eg. dota. 1000*100*30

                for _ in range(n_steps):
                    loss = s.vec_state.update(Xt, yt)
                    if loss == 0:  # gradient is 0 when loss is 0.
                        print("loss is 0, breaking early")
                        break

                s.tvec = s.vec_state.get_vec()
            elif cfg["model_type"] == "solver":
                s.tvec = adjust_vec2(s.tvec, Xt, yt, **p.solver_opts)
            else:
                assert False, "model type"
        else:
            assert False

def makeXy(idx, lr, sample_size, pseudoLabel=True):

    Xlab = idx.vectors[(lr.is_labeled > 0) ]
    ylab = lr.labels[(lr.is_labeled > 0) ]
    
    rsize = sample_size - Xlab.shape[0]

    scores = lr.current_scores()
    rsample = np.random.permutation(idx.vectors.shape[0])[:rsize]

    if pseudoLabel:

        Xsamp = idx.vectors[rsample]
        ysamp = scores[rsample]
        is_real = np.zeros(ylab.shape[0] + ysamp.shape[0])
        is_real[:ylab.shape[0]] = 1
        
        X = np.concatenate((Xlab, Xsamp))
        y = np.concatenate((ylab, ysamp))
        # if quantile_transform:
        #     ls = QuantileTransformer()
        #     ls.fit(scores.reshape(-1,1))
        #     y = ls.transform(y.reshape(-1,1)).reshape(-1)
    else:
        X = Xlab
        y = ylab
        is_real = np.ones_like(y)
        
    return X,y,is_real


class PseudoLabelLR(PointBased):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams):
        super().__init__(gdm, q, params)
        self.options = self.params.interactive_options
        self.label_prop_params = self.options['label_prop_params']
        self.log_reg_params = self.options['log_reg_params']
        self.switch_over = self.options.get('switch_over', False)
        self.real_sample_weight = self.options['real_sample_weight']
        assert self.real_sample_weight >= 1.

        knng_path = q.index.get_knng_path(self.label_prop_params['knn_path'])
        knng = KNNGraph.from_file(knng_path)
        self.knng_sym = knng.restrict_k(k=self.label_prop_params['knn_k'])
        label_prop = LabelPropagationRanker2(knng=self.knng_sym, **self.label_prop_params)

        self.knn_based = KnnBased(gdm, q, params, knn_model = label_prop)

    def set_text_vec(self, tvec):
        super().set_text_vec(tvec)
        self.knn_based.set_text_vec(tvec)

    def refine(self):
        self.knn_based.refine() # label prop
        X, y, is_real = makeXy(self.index, self.knn_based.state.knn_model, sample_size=self.options['sample_size'])
        model = LogisticRegressionPT(regularizer_vector=self.state.tvec,  **self.log_reg_params)

        weights = np.ones_like(y)
        weights[is_real > 0] = self.real_sample_weight

        model.fit(X, y.reshape(-1,1), weights.reshape(-1,1)) # y 
        self.curr_vec = model.get_coeff().reshape(-1)

    def next_batch(self):
        pos, neg = self.q.getXy(get_positions=True)

        if self.switch_over:
            if (len(pos) == 0 or len(neg) == 0):
                print('not switching over yet')
                return self.knn_based.next_batch() # tree based 
            else:
                return super().next_batch() # point based result
        else:
            return super().next_batch()



class KnnBased(LoopBase):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams, knn_model):
        super().__init__(gdm, q, params)
        self.state.knn_model = knn_model

    @staticmethod
    def from_params(gdm, q, p: SessionParams):
        knng_path = q.index.get_knng_path(name=p.interactive_options.get('knn_path', ''))
        knng = KNNGraph.from_file(knng_path)
        knng = knng.restrict_k(k=p.interactive_options['knn_k'])

        assert q.index.vectors.shape[0] == knng.nvecs

        if p.interactive == 'knn_greedy':
            knn_model = SimpleKNNRanker(knng, init_scores=None)
        elif p.interactive == 'knn_prop':
            raise ValueError('deprecated. use knn_prop2')
        elif p.interactive == 'knn_prop2':
            intra_knn_k = p.interactive_options.get('intra_knn_k', 0)
            if  intra_knn_k > 0:
                print('using composite prop')
                knng_path_frame = knng_path + '/frame_sym.parquet'
                knn_df_frame = parallel_read_parquet(knng_path_frame)
                knng_frame = KNNGraph(knn_df = knn_df_frame, nvecs=knng.nvecs)
                knng_frame = knng_frame.restrict_k(k=intra_knn_k)
            else:
                print('using simple prop')
                knng_frame = None
                # knng_frame= knng_frame.restrict_k(k=p.interactive_options['knn_k'])
            knn_model = LabelPropagationRanker2(knng_intra=knng_frame, knng=knng, **p.interactive_options)
        elif p.interactive == 'linear_prop':
            knn_model = LinearScorer(idx=self.q.index, knng_sym=knng, init_scores=None, **p.interactive_options)
        else:
            assert False

        return KnnBased(gdm, q, p, knn_model)


    def set_text_vec(self, tvec):
        scores = self.q.index.score(tvec)
        self.state.knn_model.set_base_scores(scores)


    def next_batch(self):

        """
        gets next batch of image indices based on current vector
        """
        s = self.state
        p = self.params
        q = self.q

        sorted_idxs, sorted_scores = s.knn_model.top_k(k=None, unlabeled_only=True)
        candidates = _get_top_dbidxs(vec_idxs=sorted_idxs, scores=sorted_scores, 
                        vector_meta=q.index.vector_meta, exclude=q.returned, topk=p.shortlist_size)

        raw_scores = s.knn_model.current_scores()

        frame_scores = np.zeros(candidates.shape[0])
        activations = []
        for (i,dbidx) in enumerate(candidates.dbidx.values):
            meta_df = q.index.vector_meta.query(f'dbidx == {dbidx}')
            meta_df = meta_df.assign(score=raw_scores[meta_df.index.values])
            if meta_df.shape[0] == 1:
                frame_scores[i] = meta_df.score.iloc[0]
            else:
                tup = score_frame2(meta_df, aug_larger=p.aug_larger, agg_method=p.agg_method)                
                frame_scores[i] = tup.score.iloc[0]
                activations.append(tup)

        candidates = candidates.assign(frame_scores=frame_scores)
        c = candidates.sort_values('frame_scores', ascending=False)

        idxs = c.dbidx.iloc[:p.batch_size]
        b  = {'dbidxs':idxs, 
                'scores':c.frame_scores.iloc[:p.batch_size], 
                'activations':None,
            }

        q.returned.update(idxs)        
        return b

    def refine(self):
        # labels already added.
        # go over labels here since it takes time
        ## translating box labels to labels over the vector index.
        #### for each frame in a box label. box join with the vector index for that box.
        # seen_ids = np.array(self.q.label_db.get_seen())
        pos, neg = self.q.getXy(get_positions=True)
        idxs = np.concatenate([pos,neg])
        labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        s = self.state
        s.knn_model.update(idxs, labels)


class Session:
    current_dataset: str
    current_index: str
    loop: SeesawLoop
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
        dataset: SeesawDataset,
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
        self.loop = LoopBase.from_params(self.gdm, self.q, params=self.params)
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
        p = self.loop.params
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
        for indices, accs in zip(self.acc_indices, self.acc_activations):
            imdata = self.get_panel_data(idxbatch=indices, activation_batch=accs)
            gdata.append(imdata)

        dat = {}
        dat["action_log"] = self.action_log
        dat["gdata"] = gdata
        dat["timing"] = self.timing
        dat["reference_categories"] = []
        dat["params"] = self.params
        dat["query_string"] = self.loop.state.curr_str
        return SessionState(**dat)

    def get_panel_data(self, *, idxbatch, activation_batch=None):
        reslabs = []
        #urls = get_image_paths(self.dataset.image_root, self.dataset.paths, idxbatch)
        urls = self.dataset.get_image_paths(idxbatch)

        for i, (url, dbidx) in enumerate(zip(urls, idxbatch)):
            dbidx = int(dbidx)
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
