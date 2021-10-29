import os
os.environ['OMP_NUM_THREADS'] = f'{os.cpu_count()//2}'

import flask
from flask import Flask, request
import ray
ray.init('auto', namespace='seesaw')

import seesaw
from seesaw import *
from init_data_actors import DB, RemoteDB, ModelService


gdm = GlobalDataManager('/home/gridsan/omoll/seesaw_root/data')

datasets = ['objectnet']#,[ 'dota', 'lvis','coco', 'bdd']
default_dataset = datasets[0]
dbactors = dict([(name,ray.get_actor(name)) for name in datasets])

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
        idxs, _ = ray.get(idxref)
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
    
class BoxFeedbackQueryRemote(InteractiveQueryRemote):
    def __init__(self, dbactor, batch_size, auto_fill_df=None):
        super().__init__(dbactor, batch_size)
        self.label_db = {}
        # self.rois = [[]]
        # self.augmented_rois = [[]]
        # self.roi_vecs = [np.zeros((0, dbactor..embedded.shape[1]))]
        self.auto_fill_df = auto_fill_df



def get_panel_data_remote(q, ev, label_db, next_idxs):
    reslabs = []
    for (i,dbidx) in enumerate(next_idxs):
        boxes = copy.deepcopy(label_db.get(dbidx, None))
        reslabs.append({'value': -1 if boxes is None else 1 if len(boxes) > 0 else 0, 
                        'id': i, 'dbidx': int(dbidx), 'boxes': boxes})
    urls = get_image_paths(ev, next_idxs)
    # q.db.get_image_paths(next_idxs)
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


class SessionState(object):
    ev : EvDataset
    hdb : AugmentedDB

    def __init__(self, dataset_name):
        self.current_dataset = dataset_name
        dbactor = dbactors[self.current_dataset]
        evhandle = ray.get(dbactor.get_ev.remote())
        print('getting ev state')
        ev = ray.get(evhandle)
        self.ev = ev
        print('done getting state')

        ## using coarse right now
        self.hdb = EmbeddingDB(raw_dataset=ev.image_dataset, embedding=ev.embedding,embedded_dataset=ev.embedded_dataset)
        
        # AugmentedDB(raw_dataset=ev.image_dataset, embedding=ev.embedding, 
        #     embedded_dataset=ev.fine_grained_embedding, vector_meta=ev.fine_grained_meta,
        #     vec_index=ev.vec_index)

        # if gt_class is not None:
        #     self.box_data = ray.get(self.dbactor.get_boxes.remote())
        # else:
        #     self.box_data = None
        self.box_data = None
        self.bfq = BoxFeedbackQuery(self.hdb, batch_size=10, auto_fill_df=self.box_data)
        # self.bfq = BoxFeedbackQueryRemote(self.dbactor, batch_size=10, auto_fill_df=self.box_data)
        self.init_vec = None
        self.acc_indices = np.array([])

    def reset(self, dataset_name=None):
        if dataset_name is not None:
            self.current_dataset = dataset_name
        self.dbactor = dbactors[self.current_dataset]
        self.bfq = BoxFeedbackQuery(self.hdb, batch_size=5, auto_fill_df=self.box_data)
        self.init_vec = None
        self.acc_indices = np.array([])

    def get_state(self):
        dat = get_panel_data_remote(self.bfq, self.ev,  self.bfq.label_db, self.acc_indices)
        dat['datasets'] = datasets
        dat['current_dataset'] = self.current_dataset
        return dat

    def get_latest(self):
        dat = get_panel_data_remote(self.bfq,  self.ev, self.bfq.label_db, self.bfq.acc_idxs[-1])
        return dat


state : SessionState
state = SessionState(default_dataset)

def get_image_paths(ev, idxs):
    return [ f'/data/{state.current_dataset}/images/{ev.image_dataset.paths[int(i)]}' for i in idxs]

print('inited state... about to create app')
app = Flask(__name__)

@app.route('/hello', methods=['POST'])
def hello():
    print(request.json)
    return 'hello back'

@app.route('/reset', methods=['POST'])
def reset():
    print(request.json)
    ds = request.json.get('todataset', None)
    print('reset to ', ds)
    state.reset(ds)
    return flask.jsonify(**state.get_state())

@app.route('/getstate', methods=['GET'])
def getstate():
    return flask.jsonify(**state.get_state())

@app.route('/text', methods=['POST'])
def text():
    query = request.args.get('key', '')
    state.init_vec = state.ev.embedding.from_string(string=query)
    return next()

@app.route('/search_hybrid', methods=['POST'])
def search_hybrid():
    minus_text = request.json.get('minus_text',None)
    plus_text = request.json.get('plus_text', None)
    
    normalize = lambda x : x/np.linalg.norm(x)
    standardize = lambda x : normalize(x).reshape(-1)

    image_vec = standardize(ray.get(state.dbactor.get_vectors.remote([request.json['dbidx']])))

    if minus_text is None or minus_text == '':
        minus_vec = np.zeros_like(image_vec)
    else:
        minus_vec = standardize(ray.get(state.dbactor.embed_raw.remote(minus_text)))

    if plus_text is None or plus_text == '':
        plus_vec = np.zeros_like(image_vec)
    else:
        plus_vec = standardize(ray.get(state.dbactor.embed_raw.remote(request.json['plus_text'])))

    # coeff = image_vec@minus_vec
    # cleaned_vec = image_vec - coeff*minus_vec
    # # clean_plus = plus_vec - (image_vec@plus_vec)*plus_vec
    total_vec = image_vec + plus_vec - minus_vec

    state.init_vec = total_vec.reshape(1,-1)#norm(plus_vec) + norm(image_vec) - norm(minus_vec)
    return next()

@app.route('/next', methods=['POST'])
def next():
    ldata = request.json.get('ldata', [])
    update_db(state.bfq.label_db, ldata)
    state.acc_indices = np.array([litem['dbidx'] for litem in ldata])
    acc_results = np.array([litem['value'] for litem in ldata])

    print(state.acc_indices)
    print(acc_results)
    assert state.acc_indices.shape[0] == acc_results.shape[0]
    tvec = state.init_vec
    mask = (acc_results >= 0)
    if mask.sum() > 0:
        labelled_results = acc_results[mask]
        labelled_indices = state.acc_indices[mask]
        Xt = state.ev.embedded_dataset[labelled_indices] #ray.get(state.db.get_vectors.remote(labelled_indices))
        yt = labelled_results.astype('float')

        if (yt == 0).any() and (yt == 1).any():
            tvec = update_vector(Xt,yt, state.init_vec, minibatch_size=1)

    idxbatch,_ = state.bfq.query_stateful(mode='dot', vector=tvec, batch_size=5)
    state.acc_indices = np.concatenate([state.acc_indices, idxbatch.reshape(-1)])
    dat = state.get_latest()
    # acc_results = np.array([litem['value'] for litem in dat['ldata']])
    # print('after idx', state.acc_indices)
    # print('after labels', acc_results)
    return flask.jsonify(**dat)

print('done')
if __name__ == '__main__':
    pass