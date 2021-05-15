import ray
import torch.optim
import sys
import pyroaring as pr
from ray.util.multiprocessing import Pool
import inspect
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from .cross_modal_db import *
from .embeddings import *
from .vloop_dataset_loaders import *
from .search_loop_models import *
from .search_loop_tools import *
from .ui import widgets
import logging
from tqdm.auto import tqdm
import copy

def vls_init_logger():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.captureWarnings(True)

class InteractiveQueryRemote(object):
    def __init__(self, dbactor, batch_size: int):
        self.dbactor = dbactor
        self.seen = pr.BitMap()
        self.query_history = []
        self.acc_idxs = []
        self.batch_size = batch_size

    def query_stateful(self, *args, **kwargs):
        '''
        :param kwargs: forwards arguments to db query method but
         also keeping track of already seen results. also
         keeps track of query history.
        :return:
        '''
        batch_size = kwargs.get('batch_size',self.batch_size)
        if 'batch_size' in kwargs:
            del kwargs['batch_size']
            
        idxref = self.dbactor.query.remote(*args, topk=batch_size, **kwargs, exclude=self.seen)
        idxs = ray.get(idxref)
        self.query_history.append((args, kwargs))
        self.seen.update(idxs)
        self.acc_idxs.append(idxs)
        return idxs
    
    def repeat_last(self):
        '''
        :return: continues the search from last query, effectively paging through results
        '''
        assert self.query_history != []
        args, kwargs = self.query_history[-1]
        return self.query_stateful(*args, **kwargs)
    
class BoxFeedbackQuery(InteractiveQueryRemote):
    def __init__(self, dbactor, batch_size, auto_fill_df=None):
        super().__init__(dbactor, batch_size)
        self.label_db = {}
        # self.rois = [[]]
        # self.augmented_rois = [[]]
        # self.roi_vecs = [np.zeros((0, dbactor..embedded.shape[1]))]
        self.auto_fill_df = auto_fill_df

@ray.remote(num_gpus=.2)
class DBActor(object):
    def __init__(self, dataset_loader, embedding_path, dbsample, valsample):
        # NB: in reality the n_px below may have been different for the embedding path we pass,
        # fix before using query by example. 
        tx = make_clip_transform(n_px=224, square_crop=False)
        tx2 = T.Compose([tx, lambda x : x.type(torch.float16)])
        xclip = load_clip_embedding(variant='ViT-B/32', tx=tx2, device='cuda:0', jit=True)

        vecs = np.load(embedding_path)
        ev0 = dataset_loader(xclip, embedded_vecs=vecs)

        if dbsample is not None:
            ev = extract_subset(ev0, dbsample)
            val = extract_subset(ev0, valsample)
            self.val_vec = val.db.embedded
            self.val_gt = val.query_ground_truth
        else:
            ev = ev0
            val = None
            self.val_vec = None
            self.val_gt = None

        hdb = HEmbeddingDB(ev.db)
        self.hdb = hdb
        self.ev = ev
        print('inited dbactor') 
    
    def get_urls(self, idxs):
        return [self.hdb.urls[int(dbidx)].replace('thumbnails/', '') for dbidx in idxs]

    def get_val(self):
        return (self.val_vec, self.val_gt)

    def get_qgt(self):
        return self.ev.query_ground_truth
        
    def embed_raw(self, data):
        return self.hdb.embedding.from_raw(data)
    
    def get_boxes(self, category):
        c = category
        assert category in self.ev.query_ground_truth.columns
        box_cats = self.ev.box_data.category.unique()         
        if c not in box_cats: # NB box bounds only work for bdd... fix before using example
            dbidxs = np.where(self.ev.query_ground_truth[c] > 0)[0]
            boxdf = pd.DataFrame({'dbidx':dbidxs})
            boxes = boxdf.assign(x1=0, y1=0, x2=1280, y2=720, category=c)
        else:
            boxes = self.ev.box_data[self.ev.box_data.category == c]
        return boxes
    
    def get_vectors(self, idxs=None):
        if idxs is None:
            return self.hdb.embedded
        else:
            return self.hdb.embedded[idxs]

    def query(self, *, topk, mode, cluster_id=None, vector=None, 
              model = None, exclude=None, return_scores=False):
        return self.hdb.query(topk=topk, mode=mode, cluster_id=cluster_id, 
                              vector=vector, model=model, exclude=exclude, return_scores=return_scores)



def get_panel_data_remote(q, label_db, next_idxs):
    reslabs = []
    for (i,dbidx) in enumerate(next_idxs):
        boxes = copy.deepcopy(label_db.get(dbidx, None))
        reslabs.append({'value': -1 if boxes is None else 1 if len(boxes) > 0 else 0, 
                        'id': i, 'dbidx': int(dbidx), 'boxes': boxes})
    urls = ray.get(q.dbactor.get_urls.remote(next_idxs))
    pdata = {
        'image_urls': urls,
        'ldata': reslabs,
    }
    return pdata

def make_image_panel_remote(bfq, idxbatch):
    dat = get_panel_data_remote(bfq, bfq.label_db, idxbatch)

    ldata = dat['ldata']
    if bfq.auto_fill_df is not None:
        gt_ldata = auto_fill_boxes(bfq.auto_fill_df, ldata)
        ## only use boxes for things we have not added ourselves...
        ## (ie don't overwrite db)
        for rdb, rref in zip(ldata, gt_ldata):
            if rdb['dbidx'] not in bfq.label_db:
                rdb['boxes'] = rref['boxes']

    pn = widgets.MImageGallery(**dat)
    return pn

def fit(*, mod, X, y, batch_size, valX=None, valy=None, logger=None,  max_epochs=6, gpus=0, precision=32):
    print('new fit')
    class CustomInterrupt(pl.callbacks.Callback):
        def on_keyboard_interrupt(self, trainer, pl_module):
            raise InterruptedError('custom')

    class CustomTqdm(pl.callbacks.progress.ProgressBar):
        def init_train_tqdm(self):
            """ Override this to customize the tqdm bar for training. """
            bar = tqdm(
                desc='Training',
                initial=self.train_batch_idx,
                position=(2 * self.process_position),
                disable=self.is_disabled,
                leave=False,
                dynamic_ncols=True,
                file=sys.stdout,
                smoothing=0,
                miniters=40,
            )
            return bar
    
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    
    train_ds = TensorDataset(X,torch.from_numpy(y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    if valX is not None:
        if not torch.is_tensor(valX):
            valX = torch.from_numpy(valX)
        val_ds = TensorDataset(valX, torch.from_numpy(valy))
        es = [pl.callbacks.early_stopping.EarlyStopping(monitor='AP/val', mode='max', patience=3)]
        val_loader = DataLoader(val_ds, batch_size=2000, shuffle=False, num_workers=0)
    else:
        val_loader = None
        es = []

    trainer = pl.Trainer(logger=None, 
                         gpus=gpus, precision=precision, max_epochs=max_epochs,
                         callbacks =[],
                        #  callbacks=es + [ #CustomInterrupt(),  # CustomTqdm()
                        #  ], 
                         checkpoint_callback=False,
                         progress_bar_refresh_rate=0, #=10
                        )
    trainer.fit(mod, train_loader, val_loader)


def update_vector(Xt, yt, init_vec, minibatch_size):
    p = yt.sum()/yt.shape[0]
    w = np.clip((1-p)/p, .1, 10.)
    lr = PTLogisiticRegression(Xt.shape[1], learning_rate=.0003, C=0, positive_weight=w)

    if init_vec is not None:
        iv = torch.from_numpy(init_vec)
        iv = iv / iv.norm()
        lr.linear.weight.data = iv.type(lr.linear.weight.dtype)

    fit(mod=lr, X=Xt.astype('float32'), y=yt.astype('float'), batch_size=minibatch_size)
    tvec = lr.linear.weight.detach().numpy().reshape(1,-1)   
    return tvec

default_actors = {
    'lvis':lambda : DBActor.options(name='lvis_db', num_gpus=.2, num_cpus=.1).remote(dataset_loader=lvis_full, 
                                  embedding_path='./data/coco_full_pooled_224_512_CLIP.npy',
                                  dbsample=np.load('./data/coco_30k_idxs.npy')[:10000],
                                  valsample=np.load('./data/coco_30k_idxs.npy')[10000:20000]),
    'coco':lambda : DBActor.options(name='coco_db', num_gpus=.2, num_cpus=.1).remote(dataset_loader=coco_full, 
                                  embedding_path='./data/coco_full_pooled_224_512_CLIP.npy',
                                  dbsample=np.load('./data/coco_30k_idxs.npy')[:10000],
                                  valsample=np.load('./data/coco_30k_idxs.npy')[10000:20000]),
    'dota':lambda : DBActor.options(name='dota_db', num_gpus=.2, num_cpus=.1).remote(dataset_loader=dota1_full, 
                                  embedding_path='./data/dota_224_pool_clip.npy',
                                  dbsample=np.load('./data/dota_idxs.npy')[:1000], # size is 1860 or so.
                                  valsample=np.load('./data/dota_idxs.npy')[1000:]),
    'ava': lambda : DBActor.options(name='ava_db', num_gpus=.2, num_cpus=.1).remote(dataset_loader=ava22, 
                                  embedding_path='./data/ava_dataset_embedding.npy',
                                  dbsample=np.load('./data/ava_randidx.npy')[:10000],
                                  valsample=np.load('./data/ava_randidx.npy')[10000:20000]), 
    'bdd': lambda : DBActor.options(name='bdd_db', num_gpus=.2, num_cpus=.1).remote(dataset_loader=bdd_full, 
                                  embedding_path='./data/bdd_all_valid_feats_pool_2by4_512_CLIP.npy', 
                                  dbsample=np.load('./data/bdd_20kidxs.npy')[:10000],
                                  valsample=np.load('./data/bdd_20kidxs.npy')[10000:20000]),
    'objectnet': lambda : DBActor.options(name='objectnet_db', num_gpus=.2, num_cpus=.1).remote(dataset_loader=objectnet_cropped, 
                                  embedding_path='./data/objnet_cropped_CLIP.npy', 
                                  dbsample=np.load('./data/object_random_idx.npy')[:10000],
                                  valsample=np.load('./data/object_random_idx.npy')[10000:20000])
}

if __name__ == '__main__':
    ray.init('auto')
    actors = []
    for (k,v) in default_actors.items():
        print('init ', k)
        actors.append(v())

    input('press any key to terminate the db server: ')
    print('done!')

