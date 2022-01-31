from seesaw.multiscale_index import MultiscaleIndex
import torch.nn as nn
import torch.nn.functional as F
import clip
from .dataset_search_terms import category2query
import numpy as np
import pandas as pd
from .query_interface import AccessMethod
from IPython.display import display
import copy

def load_model(device):
    variant ="ViT-B/32"
    model,_ = clip.load(variant, device=device,  jit=False)
    return model

class StringEncoder(object):
    def __init__(self, device):
        self.device = device #next(iter(clean_weights.items()))[1].device
        model = load_model(device)
        self.model = model
        # self.original_weights = copy.deepcopy(model.state_dict())
        # self.reset() # dont update the original model
        
    def encode_string(self, string):
        model = self.model.eval()
        with torch.no_grad():
            ttext = clip.tokenize([string])
            text_features = model.encode_text(ttext.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.detach().cpu().numpy()
    # def reset(self):
    #     self.model.load_state_dict(copy.deepcopy(self.original_weights))
# class StringEncoder2:
#     def __init__(self, updater : Updater2):
#         self.updater = updater
        
#     def encode_string(self, string):
#         model = self.model.eval()
#         with torch.no_grad():
#             ttext = clip.tokenize([string])
#             text_features = model.encode_text(ttext.to(self.device))
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             return text_features.detach().cpu().numpy()
from .clip_module import CLIPFineTunedModel, CLIPTx, MappedDataset
import torch
import os
import pytorch_lightning as pl

class FineTuneDataModule(pl.LightningDataModule):
    def __init__(self, image_vectors, strings, processor, *,  test_size, batch_size, val_batch_size, num_workers):
        super().__init__()
        self.processor = processor
        self.image_vectors = image_vectors
        self.strings = strings
        self.preproc = CLIPTx(processor)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(MappedDataset(self.dataset['train'], self.preproc.preprocess_tx),
                                             batch_size=self.batch_size, 
                                           shuffle=True, num_workers=self.num_workers, collate_fn=self.preproc.pad_collate)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(MappedDataset(self.dataset['test'], self.preproc.preprocess_tx), 
                                            batch_size=self.val_batch_size, 
                                           shuffle=True, num_workers=self.num_workers, collate_fn=self.preproc.pad_collate)


import pyarrow as pa
from datasets import Dataset

class OnlineModel:
  def __init__(self, processor, model, config):
    self.processor = processor
    self.config = config
    self.fine_tuner = CLIPFineTunedModel(model, **config)
    
  def encode_string(self, arg):
    self.fine_tuner.eval()
    with torch.no_grad():
      strs = self.processor.tokenizer(arg, return_tensors='pt', padding=True)
      text_features =  self.fine_tuner.model.get_text_features(**strs)
      return F.normalize(text_features).numpy()

  def update(self, imagevecs, actual_strings, target_string):
    config = self.config
    if 'SLURM_NTASKS' in os.environ:
        del os.environ["SLURM_NTASKS"]
        
    if 'SLURM_JOB_NAME' in os.environ:
        del os.environ["SLURM_JOB_NAME"]
    
    tab = pa.Table.from_pydict(mapping={'image_features':list(imagevecs), 
                                    'strings':list(actual_strings)})
    
    dataset = Dataset(tab)

    def txfun(tup):
      out = {**tup}
      toks = self.proc.tokenizer(tup['strings'], return_tensors='pt')
      out.update(toks)
      return out

    md = MappedDataset(dataset).map(txfun)

    def collate_fn(tup_list):
      token_dict = {'input_ids':[d['input_ids'][0] for d in tup_list],
                    'attention_mask':[d['attention_mask'][0] for d in tup_list]}

      ans = self.processor.tokenizer.pad(token_dict, padding='longest', return_tensors='pt')
      for k in tup_list[0].keys():
          ans[k] = torch.cat([d[k] for d in tup_list])
      return ans

    dl = torch.utils.data.DataLoader(md,
                                      batch_size=self.batch_size, 
                        shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    trainer = pl.Trainer(
        logger=None,
        num_sanity_val_steps=0,
        max_epochs=self.config['max_epochs'],
        gpus=None,
        progress_bar_refresh_rate=0,
        log_every_n_steps=0,
        callbacks=[])
    
    trainer.fit(self.fine_tuner, train_dataloader=dl)



def get_text_features(self, actual_strings, target_string):        
    s2id = {}
    sids = []
    s2id[target_string] = 0
    for s in actual_strings:
        if s not in s2id:
            s2id[s] = len(s2id)

        sids.append(s2id[s])

    strings = [target_string] + actual_strings
    ustrings = list(s2id)
    stringids = torch.tensor([s2id[s] for s in actual_strings], dtype=torch.long).to(self.device)
    tstrings = clip.tokenize(ustrings)
    text_features = self.model.encode_text(tstrings.to(self.device))
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, stringids, ustrings
        
def forward(self, imagevecs, actual_strings, target_string):
    ## uniquify strings    
    text_features, stringids, ustrings = get_text_features(self, actual_strings, target_string)
    text_features = text_features.to(self.device)
#    image_features = torch.from_numpy(imagevecs).type(text_features.dtype)
#    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#    image_features = image_features.to(self.device)
    image_features = imagevecs
    scores = image_features @ text_features.t()
    
    assert scores.shape[0] == stringids.shape[0]
    return scores, stringids.to(self.device), ustrings

def forward2(self, imagevecs, actual_strings, target_string):
    text_features, stringids, ustrings = get_text_features(self, actual_strings, target_string)
    actual_vecs = text_features[stringids]
    sought_vec = text_features[0].reshape(1,-1)
    
    image_features = torch.from_numpy(imagevecs).type(text_features.dtype)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.to(self.device)

    search_score = image_features @ sought_vec.reshape(-1)
    confounder_score = (image_features * actual_vecs).sum(dim=1)
    return search_score, confounder_score
    
import torch.optim
from .clip_module import configure_optimizer

_config = {'batch_size': 64,
 'logit_scale_init': 3.7,
 'opt_config': {'logit_scale': {'lr': 0.0001415583047102676,
   'weight_decay': 0.0017007389655182095},
  'text_model': None,
  'text_model.embeddings': None,
  'text_model.encoder.layers.0.layer_norm': {'lr': 0.0007435612322566577,
   'weight_decay': 1.5959136512232553e-05},
  'text_model.encoder.layers.11': {'lr': 0.0001298217305130271,
   'weight_decay': 0.015548602355938877},
  'text_model.encoder.layers.11.mlp': {'lr': 3.258792283209162e-07,
   'weight_decay': 0.001607367028678558},
  'text_model.encoder.layers.11.layer_norm2': None,
  'text_model.final_layer_norm': {'lr': 0.007707377565843718,
   'weight_decay': 0.0},
  'text_projection': {'lr': 2.581683501371101e-05, 'weight_decay': 0.0},
  'vision_model': None,
  'visual_projection': None},
 'num_warmup_steps': 20,
 'num_workers': 20,
 'test_size': 1000,
 'val_batch_size': 500}


class Updater(object):
    se : StringEncoder
    def __init__(self, se, config):
        self.se = se
        self.config = config
        r = configure_optimizer(self.se.model, h=config)
        self.opt = r['optimizer']
        self.lr_scheduler = r['lr_scheduler']
        self.losses = []
        
    def update(self, imagevecs, actual_strings, target_string):
        se = self.se
        se.model.train()
        opt = self.opt

        imagevecs = torch.from_numpy(imagevecs).to(se.device)

        def opt_closure():
            scores, stringids, rawstrs = forward(se, imagevecs, actual_strings, target_string)
            iidx = torch.arange(scores.shape[0]).long()
            assert scores.shape[0] == imagevecs.shape[0]
            actual_score = scores[iidx, stringids].reshape(-1,1)
            losses = F.relu(- (actual_score - scores - self.config['margin']))
            loss = losses.mean()
            return loss

        losses = []
        for _ in range(self.config['rounds']):
            opt.zero_grad()
            loss = opt_closure()
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            opt.step()
            self.lr_scheduler.step()

        self.losses.append(losses)
        return losses

from .multiscale_index import box_iou

def labelvecs(boxes, meta, iou_cutoff=.001):
    ious = box_iou(boxes, meta)
    
    maxiou = np.max(ious, axis=1)
    argmaxiou = np.argmax(ious, axis=1)
    posn = np.arange(maxiou.shape[0])
    
    boxid = posn[maxiou > iou_cutoff]
    vecid = argmaxiou[maxiou > iou_cutoff]
    cats = list(boxes.iloc[boxid].category.values)
    absvecs = meta.index.values[vecid]
    return absvecs, cats

from .multiscale_index import get_boxes, add_iou_score

def join_vecs2annotations(db : MultiscaleIndex, dbidx, annotations):
    meta = db.get_data(dbidx) 
    meta = add_iou_score(meta, annotations)
    meta = meta.assign(descriptions=meta.best_box_idx.map(lambda idx : annotations[idx].description))
    return meta

def get_box_labels(db : MultiscaleIndex, box_data, allidxs):
    vecposns = []
    astrs = []

    for dbidx in allidxs:
      meta = db.get_data(dbidx) 
      boxes = box_data[box_data.dbidx == dbidx]

      absvecs, cats = labelvecs(boxes, meta)
      vecposns.append(absvecs)
      strs = [category2query('lvis',c) for c in cats]
      astrs.extend(strs)
        
    avecids = np.concatenate(vecposns) 
    return avecids, astrs


def get_box_labels(evfull, allidxs):
    relboxes = evfull.box_data[evfull.box_data.dbidx.isin(allidxs)]
    
    vec_meta = evfull.fine_grained_meta[evfull.fine_grained_meta.dbidx.isin(allidxs)]
    vec_meta = vec_meta.drop_duplicates()
    vec_meta = vec_meta.assign(**get_boxes(vec_meta))
    
    vecposns = []
    astrs = []
    for (idx, boxes) in  relboxes.sort_values('dbidx').groupby('dbidx'):
        meta = vec_meta[vec_meta.dbidx == idx]
        absvecs, cats = labelvecs(boxes, meta)
        vecposns.append(absvecs)
        strs = [category2query('lvis',c) for c in cats]
        astrs.extend(strs)
        
    avecids = np.concatenate(vecposns) 
    return avecids, astrs

def show_scores(se, vecs, actual_strings, target_string):
    with torch.no_grad():
        se.model.eval()
        scs,stids,rawstrs = forward(se, vecs, actual_strings, target_string=target_string)
    scdf = pd.DataFrame({st:col  for  st,col in zip(rawstrs,scs.cpu().numpy().transpose())})
    display(scdf.style.highlight_max(axis=1))


#vecs = ev0.embedded_dataset
def lvisloop(ev0, se, category, firsts, max_n, batch_size, tqdm_disabled, feedback=True):
#     batch_size = 10
    n_batches = (int(max_n) // batch_size) + 1
    #tqdm_disabled = True
    sq = category2query('lvis', category)#, category)
    up = Updater(se, lr=.0001, rounds=1)

    ev, class_idxs = get_class_ev(ev0, category, boxes=True)
    evfull = extract_subset(ev0, categories=None, idxsample=class_idxs, boxes=True)
    
    dfds =  DataFrameDataset(ev.box_data[ev.box_data.category == category], 
                             index_var='dbidx', max_idx=class_idxs.shape[0]-1)
    rsz = resize_to_grid(224)
    ds = TxDataset(dfds, tx=lambda tup : rsz(im=None, boxes=tup)[1])
    imds = TxDataset(ev.image_dataset, tx = lambda im : rsz(im=im, boxes=None)[0])

    vec_meta = ev.fine_grained_meta
    vecs = ev.fine_grained_embedding
    #index_path = './data/bdd_10k_allgrains_index.ann'
    index_path = None
    hdb = AugmentedDB(raw_dataset=ev.image_dataset, embedding=ev.embedding, 
        embedded_dataset=vecs, vector_meta=vec_meta, index_path=index_path)
    
    bfq = BoxFeedbackQuery(hdb, batch_size=batch_size, auto_fill_df=None)
    rarr = ev.query_ground_truth[category]
    
    accidxs = []
    accstrs = []
    accvecids = []
#    accvecs = []
    gts = []
    for i in tqdm(range(n_batches), leave=False, disable=tqdm_disabled):
        tvec = se.encode_string(sq)
        idxbatch, other = bfq.query_stateful(mode='dot', vector=tvec, batch_size=batch_size)
        accidxs.append(idxbatch)
        gt = ev.query_ground_truth[category].iloc[idxbatch].values
        gts.append(gt)
        if gt.sum() > 0 or len(gts)*batch_size > max_n:
            break

        if feedback:
            vecids, astrs = get_box_labels(evfull, accidxs[-1])
            accvecids.append(vecids)
            accstrs.extend(astrs)

            def feedback_closure():
                avecs = evfull.fine_grained_embedding[np.concatenate(accvecids)]
                scs = avecs @ tvec.reshape(-1)
                topk = np.argsort(-scs)[:1000]
                trvecs = torch.from_numpy(avecs[topk]).float().to(se.device)
                trstrs = [accstrs[i] for i in topk]
                up.update(trvecs, trstrs, sq)
                del trvecs
                torch.cuda.empty_cache()
                
    
            feedback_closure()
        
    frsts = np.where(np.concatenate(gts).reshape(-1))[0]
    if frsts.shape[0] == 0:
        firsts[category] = np.inf
    else:
        firsts[category] = frsts[0] + 1
