from seesaw.knn_graph import KNNGraph
from .loop_base import *
from ..dataset_manager import GlobalDataManager
import numpy as np
from ..research.knn_methods import LabelPropagation
from ..research.active_search.cost_effective_active_search import  min_expected_cost_approx
from ..research.active_search.efficient_nonmyopic_search import efficient_nonmyopic_search
from ..research.active_search.common import ProbabilityModel, Dataset
from .graph_based import get_label_prop
from .LKNN_model import LKNNModel
## 1. need a loop base impl. to plug into system.
## 2. need a label prop internally for ceas model.

## planning stage: run in order to find best next position.
###  internally involves simulation of outcomes
#    not ok for it to mutate things. takes current state as starting point.
#     insert ceas code for this.
## update stage: run as a reaction to outcomes, ok for it to mutate things. re-use existing code.

### note the 3 planning parameters
### currently we seem to not be distinguishing first and last
### maximum number of rounds into the future (would be about 100)
### number of results wanted (eg 10)
### number of exact planning rounds (can only really be 1 or 2). 
import scipy.sparse as sp
from ..calibration import GroundTruthCalibrator, FixedCalibrator
from .LKNN_model import initial_gamma_array
import math

class ActiveSearch(LoopBase):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams, weight_matrix : sp.csr_array):
        super().__init__(gdm, q, params)
        self.scores = None
        dataset = Dataset.from_vectors(q.index.vectors)


        '''
        - gamma:
            mode: clip
            calibration: sigmoid | raw | grount_truth
            a: 1.
            b: 0.
        '''
        self.gamma = params.interactive_options['gamma']
        if self.gamma['mode'] == 'clip':
            calibration = self.gamma['calibration']
            if calibration == 'ground_truth':
                self._calibrator = q.get_calibrator()
                assert self._calibrator is not None
            elif calibration == 'sigmoid':
                self._calibrator = FixedCalibrator(a=self.gamma['a'], b=self.gamma['b'], sigmoid=True)
            elif calibration == 'raw':
                self._calibrator = FixedCalibrator(a=1., b=0., sigmoid=False)
            else:
                assert False, f'unknown {calibration=}'


            initial_gamma=initial_gamma_array(.1, q.index.vectors.shape[0]) # gets over-written. 

        elif self.gamma['mode'] == 'fixed':            
            initial_gamma=initial_gamma_array(self.gamma['value'], q.index.vectors.shape[0])


        self.prob_model = LKNNModel.from_dataset(dataset, gamma=initial_gamma, weight_matrix=weight_matrix)
        self.dataset = self.prob_model.dataset
        self.pruned_fractions = []
        self.refine_not_called_before = True

    @staticmethod
    def from_params(gdm, q, p: SessionParams):
        label_prop2 = get_label_prop(q, p.interactive_options)

        return ActiveSearch(gdm, q, p, weight_matrix=label_prop2.lp.weight_matrix)

    def set_text_vec(self, tvec):
        super().set_text_vec(tvec)
        self.scores = self.q.index.score(tvec)
        
        if self.gamma['mode'] == 'clip':
            probs = self._calibrator.get_probabilities(tvec, self.q.index.vectors)
            self.prob_model = self.prob_model.with_gamma(probs)
        else:
            pass

    def get_stats(self):
        return {'pruned_fractions':self.pruned_fractions}

    def next_batch(self):
        """
        gets next batch of image indices based on current vector
        """

        reward_horizon = self.params.interactive_options['reward_horizon'] 
        if self.params.interactive_options['adjust_horizon']:
            remaining_steps = self.params.interactive_options['max_steps'] - len(self.q.returned)
        else:
            remaining_steps = math.inf

        adjusted_horizon = int(min(reward_horizon, remaining_steps))
        assert adjusted_horizon > 0, f'need a non-negative horizon for reward to be defined {self.params.interactive_options["reward_horizon"]=} {remaining_steps=}'

        lookahead = min(2, adjusted_horizon) # 1 when time horizon is also 1
        res = efficient_nonmyopic_search(self.prob_model,reward_horizon=adjusted_horizon, 
                                            lookahead_limit=lookahead, 
                                            pruning_on=self.params.interactive_options['pruning_on'], 
                                            implementation=self.params.interactive_options['implementation'])
        top_idx = int(res.index) 
        print(f'{res.index=}, {res.value=}')
        self.pruned_fractions.append(res.pruned_fraction)

        vec_idx = np.array([top_idx])
        abs_idx = self.q.index.vector_meta['dbidx'].iloc[vec_idx].values
        ans = {'dbidxs': abs_idx, 'activations': None }
        self.q.returned.update(ans['dbidxs'])
        return ans

    def refine(self, change=None):
        # labels already added.
        # go over labels here since it takes time
        ## translating box labels to labels over the vector index.
        #### for each frame in a box label. box join with the vector index for that box.
        # seen_ids = np.array(self.q.label_db.get_seen())
        if change is None:
            assert False
            print(f'no change provided, need to compute from scratch')
            pos, neg = self.q.getXy(get_positions=True)
            idxs = np.concatenate([pos,neg])
            labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
            self.prob_model = self.prob_model.with_label(idxs[0], y=labels[0])
            self.dataset = self.prob_model.dataset
        else:                
            print(f'updating model with {change=}')

            translated_change = [] # only for current

            if self.refine_not_called_before:
                ## getXy includes this latest update, ignore
                ## first time. must update the db with full results.
                pos, neg = self.q.getXy(get_positions=True)
                for idx in pos:
                    translated_change.append((idx,1))
                for idx in neg:
                    translated_change.append((idx,0))
            else:
                for (idx, y) in change:
                    df = self.q.index.vector_meta
                    idx2 = df.query(f'dbidx == {idx}').index[0]
                    assert df.iloc[idx2].dbidx == idx
                    translated_change.append((idx2, y))

            for (idx, y) in translated_change:
                self.prob_model.condition_(idx,y)   

            self.refine_not_called_before = False

import sys

class LKNNSearch(LoopBase):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams, weight_matrix : sp.csr_array):
        super().__init__(gdm, q, params)

        dataset = Dataset.from_vectors(q.index.vectors)

        self._calibrator = q.get_calibrator() # only meant to be used for debugs/experiments etc. 
        if params.interactive_options['gamma'] == 'calibrate':
            assert self._calibrator is not None
            gamma_mean = self._calibrator.get_mean()
        else:
            gamma_mean = params.interactive_options['gamma']

        initial_gamma=initial_gamma_array(gamma_mean, q.index.vectors.shape[0])

        self.use_clip_as_gamma = self.params.interactive_options['use_clip_as_gamma']      
        self.prob_model = LKNNModel.from_dataset(dataset, gamma=initial_gamma, weight_matrix=weight_matrix)
        self.dataset = self.prob_model.dataset

    @staticmethod
    def from_params(gdm, q, p: SessionParams):
        label_prop2 = get_label_prop(q, p.interactive_options)
        return LKNNSearch(gdm, q, p, weight_matrix=label_prop2.lp.weight_matrix)

    def set_text_vec(self, tvec):
        super().set_text_vec(tvec)
        self.scores = self.q.index.score(tvec)

        if self.use_clip_as_gamma:
            if self._calibrator is None:
                probs = self.scores
            else:
                probs = self._calibrator.get_probabilities(tvec, self.q.index.vectors)
            
            self.prob_model = self.prob_model.with_gamma(probs)

    def next_batch(self):
        """
        gets next batch of image indices based on current vector
        """
        ### run planning stuff here. what do we do about rest of things in the frame?
        ### for now, nothing. just return one thing.
        ## 1. current scores are already propagating, no?
        vec_idx, _ = self.prob_model.top_k_remaining(top_k=1)
        print(f'{vec_idx=}')
        abs_idx = self.q.index.vector_meta['dbidx'].iloc[vec_idx].values
        ans = {'dbidxs': abs_idx, 'activations': None }
        self.q.returned.update(ans['dbidxs'])
        return ans

    def refine(self, change=None):
        # labels already added.
        # go over labels here since it takes time
        ## translating box labels to labels over the vector index.
        #### for each frame in a box label. box join with the vector index for that box.
        # seen_ids = np.array(self.q.label_db.get_seen())
        if change is None:
            assert False
        else:
            print(f'updating model with {change=}')
            for (idx, y) in change:
                df = self.q.index.vector_meta
                idx2 = df.query(f'dbidx == {idx}').index[0]
                assert df.iloc[idx2].dbidx == idx
                self.prob_model.condition_(idx2, y)

