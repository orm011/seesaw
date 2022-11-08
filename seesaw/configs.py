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
    'knn_greedy':{
        "knn_k": 5
    },
    'knn_prop' : {
        'knn_k': 5,
        'calib_a': 10.,
        'calib_b': -5,
        'prior_weight': 1.,
        'edist': .1,
        'num_iters': 5,
    }
}

def get_default_config(method):
    return _method_configs.get(method)

std_textual_config = _method_configs["textual"]
std_linear_config = _method_configs["pytorch"]

modes = { ## change terminology?
    'default':'plain', # no feedback
    # 'pytorch':'pytorch',
}

import yaml

def make_session_params(mode, dataset, index):
    _mode = modes.get(mode, mode)
    cfg = _method_configs[_mode]

    return SessionParams(
        index_spec={"d_name": dataset, "i_name": index},
        interactive=_mode,
        interactive_options=cfg, # TODO same as method config
        method_config=cfg,
        agg_method="avg_score",
        aug_larger='all',
        shortlist_size = 40,
        batch_size=3,
    )

import copy
def get_session_params(s_template, config, index_meta):
    ''' meant to be shared between benchmark code and server code. '''
    s_template = copy.deepcopy(s_template)
    config = copy.deepcopy(config)
    s_merged = {**s_template,   **config}

    prev_index_meta = s_merged.get('index_spec', {})
    final_index_meta = {
        **prev_index_meta,  # eg has index info
        **index_meta, # eg has category and dataset info
    }
    s_merged['index_spec'] = final_index_meta
    s = SessionParams(
        **{k:v for (k,v) in s_merged.items()
                if k in SessionParams.__fields__.keys()
        }
    )

    return s

import copy
from collections import namedtuple
import math
import random

def space_size(base_config):
    szs = []
    for _,v in base_config.items():
        if isinstance(v, dict) and 'choose' in v.keys() and isinstance(v['choose'], list):
            assert len(v) == 1
            szs.append(len(v['choose']))
        elif isinstance(v, dict):
            szs.append(space_size(v))
        else:
            szs.append(1)
            
    return math.prod(szs)

def sample_config(base_config):
    T = namedtuple('T', field_names=base_config.keys())

    cfg = {}
    for k,v in base_config.items():
        if isinstance(v, dict) and 'choose' in v.keys():
            assert isinstance(v['choose'], list)
            assert len(v) == 1 
            ret = random.choice(v['choose'])
            cfg[k] = ret
        elif isinstance(v, dict):
            cfg[k] = sample_config(v)
        else:
            cfg[k] = v # simply copy over

    return T(**cfg)


def asdict(t):
    base_dict = t._asdict().copy()
    for k,v in base_dict.items():
        if hasattr(v, '_asdict'):
            base_dict[k] = asdict(v)

    return base_dict

def generate_method_configs(base_config, max_trials):
    seen_configs = set()
    total_configs = space_size(base_config)
    limit = min(max_trials, total_configs)

    while len(seen_configs) < limit:
        cfg = sample_config(base_config)
        seen_configs.add(cfg)

    ans = []
    for i,cfgelt in enumerate(seen_configs):
        cfg = asdict(cfgelt)
        if len(seen_configs) > 1:
            cfg['sample_id'] = f"sample_{i:02d}"
        ans.append(cfg)

    return ans

def expand_configs(variants):
    expanded_configs = []
    for var in variants:
        gconfigs = generate_method_configs(var, max_trials=var.get('max_samples', 1))
        expanded_configs.extend(gconfigs)

    return expanded_configs