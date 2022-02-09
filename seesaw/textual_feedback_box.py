from seesaw.multiscale_index import MultiscaleIndex
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import pandas as pd
from clip.model import build_model
from .clip_module import CLIPFineTunedModel, CLIPTx, MappedDataset
import torch
from .multiscale_index import add_iou_score
from ray.data.extensions import TensorArray
import torch.optim
from .clip_module import configure_optimizer


def join_vecs2annotations(db : MultiscaleIndex, dbidx, annotations):
    patch_box_df = db.get_data(dbidx)
    roi_box_df = pd.DataFrame.from_records([b.dict() for b in annotations])

    dfvec = add_iou_score(patch_box_df, roi_box_df)
    dfvec = dfvec.assign(descriptions=dfvec.best_box_idx.map(lambda idx : annotations[idx].description))
  
    dfbox = add_iou_score(roi_box_df, patch_box_df)

    matched_vecs = np.stack([ dfvec.vectors.iloc[i].copy()  for i in dfbox.best_box_idx.values])
    dfbox = dfbox.assign(descriptions=dfbox.description, vectors=TensorArray(matched_vecs))

    return dfvec, dfbox

def deduplicate_strings(string_list):
    s2id = {}
    sids = []
    for s in string_list:
        if s not in s2id: # add new string to dict
            s2id[s] = len(s2id)

        sids.append(s2id[s])
    
    string_ids = np.array(sids)
    
    reverse_dict = {num:strng for (strng,num) in s2id.items()}
    id2string = np.array([reverse_dict[i] for i in range(len(s2id))])

    return {'strings':id2string, 'indices':string_ids}
            
# using open ai clip model param names
std_textual_config = {'batch_size': 64,
                  'logit_scale_init': 3.7,
                  'image_loss_weight':1., # 
                  'vector_box_min_iou':.2, # when matching vectors to user boxes what to use
                  'device':'cuda:0',
                  'opt_config': {'logit_scale': None, #{'lr': 0.0001415583047102676,'weight_decay': 0.0017007389655182095},
                    'transformer': None,
                    'transformer.resblocks.0.ln_': {'lr': 0.0007435612322566577,'weight_decay': 1.5959136512232553e-05},
                    'transformer.resblocks.11.ln': {'lr': 0.0001298217305130271,'weight_decay': 0.015548602355938877},
                    #'transformer.resblocks.11.mlp': None, #{'lr': 3.258792283209162e-07,'weight_decay': 0.001607367028678558},
                    #'transformer.resblocks.11.ln_2': None,
                    'ln_final': {'lr': 0.007707377565843718,'weight_decay': 0.0},
                    'text_projection': {'lr': 5.581683501371101e-05, 'weight_decay': 0.0},
                    'positional_embedding':None,
                    'token_embedding':None,
                    'visual': None,
                    'positiional_embedding':None},
                  'num_warmup_steps': 5,
                  'rounds': 4,
                  'margin':.3,
                  'num_workers': 20,
                  'test_size': 1000,
                  'val_batch_size': 500}


class OnlineModel:
    def __init__(self, state_dict, config):
      self.original_weights = state_dict
      if not torch.cuda.is_available():
        if config['device'].startswith('cuda'):
          print('Warning: no GPU available, using cpu instead')
          config = {**config, 'device':'cpu'} # overrule gpu if not available

      self.device = config['device']
      self.model = None
      self.config = config
      self._reset_model()
      self.losses = []

    def _reset_model(self): # resets model and optimizers
      print('resetting model state')
      mod =  build_model(self.original_weights).float()
      self.model = mod.to(self.device)
      print('done resetting model')

    def _encode_string(self, tokenized_strings):
        text_features = self.model.encode_text(tokenized_strings.to(self.device))
        text_features  = F.normalize(text_features)
        return text_features

    def encode_string(self, string) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tokens = clip.tokenize([string])
            vecs = self._encode_string(tokens)
            return vecs.cpu().numpy()

        
    def update(self, imagevecs, marked_accepted, annotations, target_string):
        self._reset_model()
        self.model.train()

        assert imagevecs.shape[0] == marked_accepted.shape[0]
        assert annotations.shape[0] == marked_accepted.shape[0]

        r = configure_optimizer(self.model, self.config)
        opt : torch.optim.Optimizer = r['optimizer']
        lr_scheduler = r['lr_scheduler']

        imagevecs = torch.from_numpy(imagevecs).to(self.device)

        d = deduplicate_strings([target_string] + list(annotations))
        strings = d['strings']
        string_ids = d['indices']


        annotation_ids = string_ids[1:] # first string is the target string
        orig_strings = strings[annotation_ids]
        ## want mask for all vectors where the user has provided a better description
        ## ie. exclude empty descriptions '', and also exclude cases where the description 
        ## is identical to the search query used as reference (we would be penalizing it)
        better_described = (orig_strings != '') & (orig_strings != target_string)

        total_better_described = better_described.sum()
        total_positive = marked_accepted.sum()
        print(f'starting update: total better described: {total_better_described}. total positive/negative : {total_positive}/{(~marked_accepted).sum()}')

        if total_better_described == 0 \
            and not (0 < total_positive < marked_accepted.shape[0]):
              print('need at least some annotations for which a loss can be computed')
              return 0.# no annotations can be used yet (loss is 0)        

        tokens = clip.tokenize(strings)
        def opt_closure():
            text_features = self._encode_string(tokens)
            scores = imagevecs @ text_features.t()
            score_for_target_string = scores[:,0]
            score_for_annotation = scores[torch.arange(scores.shape[0]), string_ids[1:]] # picks the corresponding index for each 

            if better_described.sum() > 0:
              # the given label should score higher than the search query
              label_rank_losses = F.relu(- (score_for_annotation[better_described] - score_for_target_string[better_described] - self.config['margin']))
              label_rank_loss = label_rank_losses.mean()
            else:
              label_rank_loss = 0.

            if 0 < total_positive < marked_accepted.shape[0]:
              pos_scores = score_for_target_string[marked_accepted]
              neg_scores = score_for_target_string[~marked_accepted]

              image_rank_losses = F.relu(- (pos_scores.reshape(-1,1) - neg_scores.reshape(1,-1) - self.config['margin']))
              image_rank_loss = image_rank_losses.reshape(-1).mean()
            else:
              image_rank_loss = 0.

            print('label loss for step: ', label_rank_loss.detach().cpu().item() if torch.is_tensor(label_rank_loss) else label_rank_loss)
            print('image loss for step: ', image_rank_loss.detach().cpu().item() if torch.is_tensor(image_rank_loss) else image_rank_loss)

            loss = label_rank_loss + self.config['image_loss_weight']*image_rank_loss
            return loss

        losses = []
        for _ in range(self.config['rounds'] + self.config['num_warmup_steps']):
            opt.zero_grad()
            loss = opt_closure()
            loss.backward()
            losses.append(loss.detach().cpu().numpy().item())
            opt.step()
            lr_scheduler.step()

        self.losses.append(losses)
        return losses
