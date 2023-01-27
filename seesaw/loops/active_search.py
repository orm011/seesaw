from .loop_base import *
from ..dataset_manager import GlobalDataManager
import numpy as np
from ..research.knn_methods import LabelPropagation
from ..research.cost_effective_active_search import IncrementalModel, Dataset, min_expected_cost_approx
from .graph_based import get_label_prop

## 1. need a loop base impl. to plug into system.
## 2. need a label prop internally for ceas model.

## planning stage: run in order to find best next position.
###  internally involves simulation of outcomes
#    not ok for it to mutate things. takes current state as starting point.
#     insert ceas code for this.
## update stage: run as a reaction to outcomes, ok for it to mutate things. re-use existing code.

class ActiveSearch(LoopBase):
    def __init__(self, gdm: GlobalDataManager, q: InteractiveQuery, params: SessionParams, knn_model):
        super().__init__(gdm, q, params)
        self.state.knn_model = knn_model
        self.dataset = Dataset.from_vectors(q.index.vectors)

    @staticmethod
    def from_params(gdm, q, p: SessionParams):
        label_prop2 = get_label_prop(q, p.interactive_options)
        return ActiveSearch(gdm, q, p, knn_model=label_prop2)


    def set_text_vec(self, tvec):
        scores = self.q.index.score(tvec)
        self.state.knn_model.set_base_scores(scores)

    def next_batch(self):
        """
        gets next batch of image indices based on current vector
        """
        ### run planning stuff here. what do we do about rest of things in the frame?
        ### for now, nothing. just return one thing.

        knnm= self.state.knn_model
        prop_model = PropModel(self.dataset, knnm.lp, predicted=knnm.current_scores())

        new_r = 3
        #TODO: r should depend on configuration target  - current state?
        ## what does it mean for vectors in the same image?

        res = min_expected_cost_approx(new_r, t=1, top_k=3, model=prop_model)
        top_idx = res.index
        ans = {'dbidxs': np.array([top_idx]), 'activations': None }
        self.q.returned.update(ans['dbidxs'])
        return ans

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
        self.dataset = Dataset.from_labels(idxs, labels, self.dataset.vectors)


### how does the first lp get made? copy what you would normally use.


class PropModel(IncrementalModel):
    def __init__(self, dataset : Dataset, lp : LabelPropagation, predicted : np.ndarray):
        super().__init__(dataset)
        self.lp  = lp
        self.predicted = predicted


    def with_label(self, idx, y) -> 'PropModel':
        ''' returns new model
        '''

        new_dataset = self.dataset.with_label(idx, y)
        idxs, labs = new_dataset.get_labels()
        new_predicted = self.lp.fit_transform(label_ids=idxs, label_values=labs, start_value=self.predicted)
        return PropModel(new_dataset, self.lp, new_predicted)

    def predict_proba(self, idxs : np.ndarray ) -> np.ndarray:
        return self.predicted[idxs]

    