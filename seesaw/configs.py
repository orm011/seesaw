from .basic_types import SessionParams

_method_configs = {
    "pytorch": {
        "minibatch_size": 10,
        "learning_rate": 0.005,
        "max_examples": 500,
        "loss_margin": 0.1,
        "num_epochs": 4,
        "model_type": "cosine",
        "warm_start": "warm",
    },
    "textual": {
        "mode": "linear",  # 'finetune' or 'linear'
        "image_loss_weight": 0.5,  #
        "vector_box_min_iou": 0.2,  # when matching vectors to user boxes what to use
        "device": "cuda:0",
        "opt_config": {
            "logit_scale": None,  # {'lr': 0.0001415583047102676,'weight_decay': 0.0017007389655182095},
            "transformer": None,
            # 'transformer.resblocks.0.ln_': {'lr': 0.0007435612322566577,'weight_decay': 1.5959136512232553e-05},
            # 'transformer.resblocks.11.ln': {'lr': 0.0001298217305130271,'weight_decay': 0.015548602355938877},
            #'transformer.resblocks.11.mlp': None, #{'lr': 3.258792283209162e-07,'weight_decay': 0.001607367028678558},
            #'transformer.resblocks.11.ln_2': None,
            # 'ln_final': {'lr': 0.007707377565843718,'weight_decay': 0.0},
            "ln_final": None,
            "text_projection": {"lr": 5.581683501371101e-05, "weight_decay": 0.0},
            "positional_embedding": None,
            "token_embedding": None,
            "visual": None,
            "positiional_embedding": None,
        },
        "num_warmup_steps": 4,
        "rounds": 4,
        "label_margin": 0.1,
        "rank_margin": 0.1,
    },
    "plain": {},
}

std_textual_config = _method_configs["textual"]
std_linear_config = _method_configs["pytorch"]

_session_modes = {
    "default": SessionParams(
        index_spec={"d_name": "", "i_name": "coarse"},
        interactive="plain",
        method_config=_method_configs["plain"],
        agg_method="avg_vector",
        batch_size=3,
    ),
    "fine": SessionParams(
        index_spec={"d_name": "", "i_name": "multiscale"},
        interactive="plain",
        method_config=_method_configs["plain"],
        agg_method="avg_vector",
        batch_size=3,
    ),
    "pytorch": SessionParams(
        index_spec={"d_name": "", "i_name": "multiscale"},  ## seesaw
        interactive="pytorch",
        method_config=_method_configs["pytorch"],
        agg_method="avg_vector",
        batch_size=3,
    ),
    "textual": SessionParams(
        index_spec={"d_name": "", "i_name": "multiscale"},
        interactive="textual",
        method_config=std_textual_config,
        agg_method="avg_vector",
        batch_size=3,
    ),
}

_dataset_map = {
    "lvis": "lvis",
    "coco": "coco",
    "bdd": "bdd",
    "objectnet": "objectnet",
    "bdd_track": "bdd_track", 
    "manny_bdd_1000": "manny_bdd_100",
}

_index_map = {
    "multiscale": "multiscale", 
    "coarse": "coarse", 
    "roi": "roi", 
    "roi_track": "roi_track", 
}
