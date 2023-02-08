from seesaw.knn_graph import KNNGraph
from .loop_base import *
from ..dataset_manager import GlobalDataManager
import numpy as np
from ..research.knn_methods import LabelPropagation
from ..research.active_search.cost_effective_active_search import  min_expected_cost_approx
from ..research.active_search.efficient_nonmyopic_search import efficient_nonmyopic_search
from ..research.active_search.common import ProbabilityModel, Dataset
from .graph_based import get_label_prop

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


class ActiveSearch(LoopBase):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams, weight_matrix : sp.csr_array):
        super().__init__(gdm, q, params)
        self.scores = None

        dataset = Dataset.from_vectors(q.index.vectors)
        self.prob_model = LKNNModel.from_dataset(dataset, gamma=.1, weight_matrix=weight_matrix)
        self.dataset = self.prob_model.dataset

    @staticmethod
    def from_params(gdm, q, p: SessionParams):
        label_prop2 = get_label_prop(q, p.interactive_options)
        return ActiveSearch(gdm, q, p, weight_matrix=label_prop2.lp.weight_matrix)

    def set_text_vec(self, tvec):
        self.scores = self.q.index.score(tvec)

    def next_batch(self):
        """
        gets next batch of image indices based on current vector
        """
        ### run planning stuff here. what do we do about rest of things in the frame?
        ### for now, nothing. just return one thing.
        ## 1. current scores are already propagating, no?

        if len(self.q.returned) == 0: # return same result as clip first try.
            top_idx = np.argmax(self.scores)
        else:
            new_r = 10
            max_t = 2
            top_k = 100

            #TODO: r should depend on configuration target  - current state?
            ## what does it mean for vectors in the same image?
            #res = min_expected_cost_approx(new_r, t=max_t, top_k=None, model=prop_model)
            res = efficient_nonmyopic_search(self.prob_model, time_horizon=top_k, lookahead_limit=0, pruning_on=False)
            top_idx = res.index
        
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
            for (idx, y) in change:
                df = self.q.index.vector_meta
                idx2 = df.query(f'dbidx == {idx}').index[0]
                self.prob_model = self.prob_model.with_label(idx2, y)

### how does the first lp get made? copy what you would normally use.
class PropagationModel(ProbabilityModel):
    def __init__(self, dataset : Dataset, lp : LabelPropagation, predicted : np.ndarray):
        super().__init__(dataset)
        self.lp  = lp
        self.predicted = predicted


    def with_label(self, idx, y) -> 'PropagationModel':
        ''' returns new model
        '''

        new_dataset = self.dataset.with_label(idx, y)
        idxs, labs = new_dataset.get_labels()
        new_predicted = self.lp.fit_transform(label_ids=idxs, label_values=labs, start_value=self.predicted)
        return PropagationModel(new_dataset, self.lp, new_predicted)

    def predict_proba(self, idxs : np.ndarray ) -> np.ndarray:
        return self.predicted[idxs]
    

class LKNNModel(ProbabilityModel):
    ''' Implements L-KNN prob. model used in Active Search paper.
    '''    
    def __init__(self, dataset : Dataset, gamma : float, matrix : sp.csr_array, numerators : np.ndarray, denominators : np.ndarray):
        super().__init__(dataset)
        self.matrix = matrix
        self.numerators = numerators
        self.denominators = denominators
        self.gamma = gamma

        assert dataset.vectors.shape[0] == matrix.shape[0]
        print(f'{matrix.shape=}')

        ## set probs to estimates, then replace estimates with labels
        self._probs = (gamma + numerators) / (1 + denominators)

        if len(dataset.seen_indices) > 0:
            idxs, labels = dataset.get_labels()
            self._probs[idxs] = labels


    @staticmethod
    def from_dataset( dataset : Dataset, weight_matrix : sp.csr_array, gamma : float):
        assert weight_matrix.format == 'csr'
        assert len(dataset.idx2label) == 0, 'not implemented other case'
        ## need to initialize numerator and denominator
        sz = weight_matrix.shape[0]
        return LKNNModel(dataset, gamma=gamma, matrix=weight_matrix, numerators=np.zeros(sz), denominators=np.zeros(sz))


    def with_label(self, idx, y) -> 'LKNNModel':
        ''' returns new model
        '''

        numerators = self.numerators.copy()
        denominators = self.denominators.copy()


        row  = self.matrix.getrow(idx) # may include itself, but will ignore these
        _, neighbors = row.nonzero()
        #neighbors = neighbors.reshape(-1)
        print(neighbors)

        curr_label = self.dataset.idx2label.get(idx, None)
        if curr_label is None:
            numerators[neighbors] += y
            denominators[neighbors] += 1
        elif curr_label != y:
            numerators[neighbors] += (y - curr_label)
        else: # do nothing.
            pass

        new_dataset = self.dataset.with_label(idx, y)
        return LKNNModel(new_dataset, gamma=self.gamma, matrix=self.matrix, numerators=numerators, denominators=denominators)

    def predict_proba(self, idxs : np.ndarray ) -> np.ndarray:
        return self._probs[idxs]

    def pbound(self, n) -> np.ndarray:
        idxs = self.dataset.remaining_indices()
        prob_bounds = (self.gamma + n + self.numerators[idxs])/(1 + n + self.denominators[idxs])
        return np.max(prob_bounds)