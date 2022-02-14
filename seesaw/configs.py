from .basic_types import SessionParams
from .textual_feedback_box import std_textual_config

_session_modes = {
  'default':SessionParams(index_spec={'d_name':'', 'i_name':''},
          interactive='plain', 
          method_config=None,
          warm_start='warm', batch_size=3, 
          minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
          num_epochs=2, model_type='cosine'),
  'box':SessionParams(index_spec={'d_name':'', 'i_name':''},
          interactive='pytorch', 
          method_config=None,
          warm_start='warm', batch_size=3, 
          minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
          num_epochs=2, model_type='cosine'),
  'textual':SessionParams(index_spec={'d_name':'', 'i_name':''},
          interactive='textual', 
          method_config=std_textual_config,
          warm_start='warm', batch_size=3, 
          minibatch_size=10, learning_rate=0.01, max_examples=225, loss_margin=0.1,
          num_epochs=2, model_type='cosine'),
}

_dataset_map = {
  # 'lvis':'data/lvis/',
  'coco':'data/coco/',
  'bdd':'data/bdd/',
  'objectnet': 'data/objectnet/'
}