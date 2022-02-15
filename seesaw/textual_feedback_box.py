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
            

class LinearScorer(nn.Module):
    def __init__(self, init_weight : torch.Tensor):
      super().__init__()
      self.weight = nn.Parameter(data=init_weight, requires_grad=True)
      self.bias = nn.Parameter(torch.tensor(0., device=init_weight.device), requires_grad=True)
      self.logit_scale = nn.Parameter(torch.tensor(0., device=init_weight.device), requires_grad=True)

    def get_vec(self):
      return self.weight.detach().cpu().numpy()

    def forward(self, X : torch.Tensor):
      return (X @ self.weight.t()) * self.logit_scale.exp() + self.bias

class DynamicLinear(nn.Module):
    def __init__(self, device):
      super().__init__()
      self.device = device
      self.scorers = nn.ModuleDict()

    def add_scorer(self, name : str, init_weight : torch.Tensor):
      assert name not in self.scorers  
      self.scorers.add_module(name, LinearScorer(init_weight=init_weight).to(self.device))

    def get_vec(self, name):
      return self.scorers[name].get_vec()

    def forward(self, vecs : torch.Tensor):
      assert len(vecs.shape) == 2
      scores = []
      for (_, scorer) in self.scorers.items():
        scores.append(scorer(vecs))

      return torch.stack(scores).t()

import transformers
### the way we pick vectors for training right now should be revisited

class OnlineModel:
    def __init__(self, state_dict, config):
      if not torch.cuda.is_available():
        if config['device'].startswith('cuda'):
          print('Warning: no GPU available, using cpu instead')
          config = {**config, 'device':'cpu'} # overrule gpu if not available

      self.original_weights = {k:v.float() for (k,v) in state_dict.items()}

      self.device = config['device']
      self.model = build_model(self.original_weights).float().to(self.device)
      self.linear_scorer = None
      self.config = config
      self._reset_model()
      self.mode = self.config['mode']
      self.losses = []
      self._cache = {}
      
    def _reset_model(self): # resets model and optimizers
      print('resetting model state')
      layers = ['text_projection']
      reset_weights = {k:v.to(self.device) for (k,v) in self.original_weights.items() if k in layers}
      self.model.load_state_dict(reset_weights, strict=False)
      self.linear_scorer = DynamicLinear(self.device)
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

    def compute_up_to(self, strings, layer) -> torch.Tensor:
        assert layer == 'text_projection'

        non_cached = []
        for s in strings:
          if s not in self._cache:
            non_cached.append(s)

        def closure(self, tokens): # pass self.model
          # taken from model.encode_text
          x = self.token_embedding(tokens).type(self.dtype)  # [batch_size, n_ctx, d_model]

          x = x + self.positional_embedding.type(self.dtype)
          x = x.permute(1, 0, 2)  # NLD -> LND
          x = self.transformer(x)
          x = x.permute(1, 0, 2)  # LND -> NLD
          x = self.ln_final(x).type(self.dtype)

          # x.shape = [batch_size, n_ctx, transformer.width]
          # take features from the eot embedding (eot_token is the highest number in each sequence)
          x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
          return x

        if len(non_cached) > 0:
          tokens = clip.tokenize(non_cached).to(self.device) 
          new_vecs = closure(self.model, tokens)
          for (s,v) in zip(non_cached, new_vecs):
            self._cache[s] = v
          
        ans = []
        for s in strings:
          ans.append(self._cache[s])

        return torch.stack(ans)
        

    def compute_from(self, x, layer) -> torch.Tensor:
        assert layer == 'text_projection'

        def closure(self, x):
          x =  x @ self.text_projection
          return F.normalize(x)
        
        return closure(self.model, x)

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

        with torch.no_grad():
          self.model.eval()
          constant_activations = self.compute_up_to(strings, 'text_projection')

        def opt_closure():
            self.model.train()
            text_features = self.compute_from(constant_activations, 'text_projection') 
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
