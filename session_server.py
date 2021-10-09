import flask
from flask import Flask, request
import ray


from seesaw import *
# from .search_loop_models import *
# from .search_loop_tools import *
# from .vloop_dataset_loaders import *
# from .cross_modal_db import *
# from .embedding_plot import *
# from .embeddings import *
from init_data_actors import BoxFeedbackQueryRemote, get_panel_data_remote, update_vector


ray.init('auto', 
    #ignore_reinit_error=True,
        namespace='seesaw')
default_dataset = 'mini_coco'
datasets = ['mini_coco',] #'lvis',  ]#['coco', 'ava', 'bdd', 'dota', 'objectnet', 'lvis']
dbactors = dict([(name,ray.get_actor('{}_db'.format(name))) for name in datasets])


class SessionState(object):
    def __init__(self, dataset_name, gt_class=None):
        self.current_dataset = dataset_name
        self.dbactor = dbactors[self.current_dataset]
        if gt_class is not None:
            self.box_data = ray.get(self.dbactor.get_boxes.remote())
        else:
            self.box_data = None

        self.bfq = BoxFeedbackQueryRemote(self.dbactor, batch_size=10, auto_fill_df=self.box_data)
        self.init_vec = None
        self.acc_indices = np.array([])

    def reset(self, dataset_name=None):
        if dataset_name is not None:
            self.current_dataset = dataset_name
        self.dbactor = dbactors[self.current_dataset]
        self.bfq = BoxFeedbackQueryRemote(self.dbactor, batch_size=5, auto_fill_df=self.box_data)
        self.init_vec = None
        self.acc_indices = np.array([])

    def get_state(self):
        dat = get_panel_data_remote(self.bfq,  self.bfq.label_db, self.acc_indices)
        dat['datasets'] = datasets
        dat['current_dataset'] = self.current_dataset
        return dat

    def get_latest(self):
        dat = get_panel_data_remote(self.bfq,  self.bfq.label_db, self.bfq.acc_idxs[-1])
        return dat

state = SessionState(default_dataset)


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
    state.init_vec = ray.get(state.dbactor.embed_raw.remote(query))
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
        Xt = ray.get(state.dbactor.get_vectors.remote(labelled_indices))
        yt = labelled_results.astype('float')

        if (yt == 0).any() and (yt == 1).any():
            tvec = update_vector(Xt,yt, state.init_vec, minibatch_size=1)

    idxbatch = state.bfq.query_stateful(mode='dot', vector=tvec, batch_size=5)
    state.acc_indices = np.concatenate([state.acc_indices, idxbatch])
    dat = state.get_latest()
    # acc_results = np.array([litem['value'] for litem in dat['ldata']])
    # print('after idx', state.acc_indices)
    # print('after labels', acc_results)
    return flask.jsonify(**dat)

if __name__ == '__main__':
    pass