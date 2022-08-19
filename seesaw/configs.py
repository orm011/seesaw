from .basic_types import SessionParams

_method_configs = {
    "pytorch": {
        "minibatch_size": 1000,
        "learning_rate": 0.01,
        "max_examples": 1000,
        "loss_margin": 0.02,
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
    "plain": {"dummy":"dummy"},
}

std_textual_config = _method_configs["textual"]
std_linear_config = _method_configs["pytorch"]
std_plain_config = _method_configs["plain"]

modes = { ## change terminology?
    'default':'plain', # no feedback
    'pytorch':'pytorch',
}


def make_session_params(mode, dataset, index):
    _mode = modes[mode]
    return SessionParams(
        index_spec={"d_name": dataset, "i_name": index},
        interactive=_mode,
        method_config=_method_configs[_mode],
        agg_method="avg_score",
        aug_larger='all',
        shortlist_size = 40,
        batch_size=3,
    )