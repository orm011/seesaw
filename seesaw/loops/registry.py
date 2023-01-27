def build_loop_from_params(gdm, q, params):
    from .pseudo_lr import PseudoLR
    from .log_reg import LogReg2
    from .point_based import Plain
    from .graph_based import KnnProp2

    # from .old_seesaw import OldSeesaw
    # from .switch_over import SwitchOver
    from .active_search import ActiveSearch

    cls_dict = {
        'knn_prop2':KnnProp2,
        'plain':Plain,
        'log_reg2':LogReg2,
        'pseudo_lr':PseudoLR,
        'active_search':ActiveSearch,

        ## older, may need to implement from_params()
      #  'old_seesaw':OldSeesaw, # aka pytorch in old code
      #  'switch_over':SwitchOver,
    }

    cls = cls_dict.get(params.interactive, None)
    return cls.from_params(gdm, q, params)
