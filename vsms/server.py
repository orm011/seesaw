import flask
from flask import Flask, request
from .search_loop_models import *
from .search_loop_tools import *
from .vloop_dataset_loaders import *
from .cross_modal_db import *
from .embedding_plot import *
from .embeddings import *
import ray
from .dbserver import BoxFeedbackQuery, get_panel_data_remote, update_vector

app = Flask(__name__)
ray.init('auto', ignore_reinit_error=True)

default_dataset = 'coco'
datasets = ['coco', 'ava', 'bdd', 'dota']
dbactors = dict([(name,ray.get_actor('{}_db'.format(name))) for name in datasets])

class SessionState(object):
    def __init__(self, dataset_name):
        self.current_dataset = dataset_name
        self.dbactor = dbactors[self.current_dataset]
        self.bfq = BoxFeedbackQuery(self.dbactor, batch_size=5)
        self.init_vec = None
        self.acc_indices = np.array([])

    def reset(self, dataset_name=None):
        if dataset_name is not None:
            self.current_dataset = dataset_name
        self.dbactor = dbactors[self.current_dataset]
        self.bfq = BoxFeedbackQuery(self.dbactor, batch_size=5)
        self.init_vec = None
        self.acc_indices = np.array([])

    def get_state(self):
        dat = get_panel_data_remote(self.bfq,  self.bfq.label_db, self.acc_indices)
        dat['datasets'] = datasets
        dat['current_dataset'] = self.current_dataset
        return dat

state = SessionState(default_dataset)

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

@app.route('/categories', methods=['GET'])
def options():
    vals = list(ev.query_ground_truth.columns.values)
    return flask.jsonify(vals)

@app.route('/text', methods=['POST'])
def text():
    query = request.args.get('key', '')
    state.init_vec = ray.get(state.dbactor.embed_raw.remote(query))
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
    dat = state.get_state()
    acc_results = np.array([litem['value'] for litem in dat['ldata']])
    print('after idx', state.acc_indices)
    print('after labels', acc_results)
    return flask.jsonify(**dat)

