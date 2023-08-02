from seesaw.loops.multi_reg_neg import MultiRegNeg
from seesaw.loops.multi_reg import MultiReg
from seesaw.loops.active_search import LKNNSearch
from seesaw.loops.rocchio_update import RocchioUpdate
from seesaw.loops.random_results import RandomResults

def build_loop_from_params(gdm, q, params):
    from .pseudo_lr import PseudoLR
    from .log_reg import LogReg2
    from .point_based import Plain
    from .graph_based import KnnProp2
    from .multi_reg import MultiReg

    from .multi_reg_neg import MultiRegNeg
    # from .old_seesaw import OldSeesaw
    # from .switch_over import SwitchOver
    from .active_search import ActiveSearch, LKNNSearch

    cls_dict = {
        'knn_prop2':KnnProp2,
        'plain':Plain,
        'log_reg2':LogReg2,
        'pseudo_lr':PseudoLR,
        'active_search':ActiveSearch,
        'lknn':LKNNSearch,
        'multi_reg':MultiReg, # seesaw.
        'rocchio_update':RocchioUpdate,
        'random':RandomResults,
        'multi_reg_neg':MultiRegNeg,

        ## older, may need to implement from_params()
      #  'old_seesaw':OldSeesaw, # aka pytorch in old code
      #  'switch_over':SwitchOver,
    }

    cls = cls_dict.get(params.interactive)
    return cls.from_params(gdm, q, params)
