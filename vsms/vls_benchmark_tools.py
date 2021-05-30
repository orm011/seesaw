from .data_server import *
from .search_loop_models import *

def run_loop(*, dbactor, category, qstr, interactive, warm_start, n_batches, batch_size, minibatch_size, 
model_type='logistic', **kwargs):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        allargs = {k:v for (k,v) in values.items() if k in args}
        print('starting: ')
        print(allargs)
        
        dbvec = ray.get(dbactor.get_vectors.remote())
        boxes = ray.get(dbactor.get_boxes.remote(category))
        
        if qstr == 'nolang':
            init_vec=None
            init_mode = 'random'
        else:
            init_vec = ray.get(dbactor.embed_raw.remote(qstr))
            init_mode = 'dot'
        
        bfq = BoxFeedbackQuery(dbactor, batch_size=batch_size, auto_fill_df=boxes)
        acc_indices = []
        acc_results = []
        res = {'indices':acc_indices, 'results':acc_results}

        tvec = init_vec
        tmode = init_mode
        for i in tqdm(range(n_batches), leave=False):
            idxbatch = bfq.query_stateful(mode=tmode, vector=tvec, batch_size=batch_size)
            pn = make_image_panel_remote(bfq, idxbatch)
            ldata = pn.ldata
            update_db(bfq.label_db, ldata)
            _, flag = binary_panel_data(ldata)
            acc_indices.append(idxbatch)
            acc_results.append(flag)
            
            if interactive != 'plain':
                Xt = dbvec[np.concatenate(acc_indices)]
                yt = np.concatenate(acc_results)
                if (yt == 0).any() and (yt == 1).any():
                    tmode = 'dot'
                    if interactive == 'sklearn':
                        lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
                        lr.fit(Xt, yt)
                        tvec = lr.coef_.reshape(1,-1)        
                    elif interactive == 'pytorch':
                        p = yt.sum()/yt.shape[0]
                        w = np.clip((1-p)/p, .1, 10.)

                        if model_type == 'logistic':
                            mod = PTLogisiticRegression(Xt.shape[1], learning_rate=.0003, C=0, positive_weight=w)
                            if warm_start == 'warm':
                                iv = torch.from_numpy(init_vec)
                                iv = iv / iv.norm()
                                mod.linear.weight.data = iv.type(mod.linear.weight.dtype)
                            elif warm_start == 'default':
                                pass

                            fit_reg(mod=mod, X=Xt.astype('float32'), y=yt.astype('float'), batch_size=minibatch_size)
                            tvec = mod.linear.weight.detach().numpy().reshape(1,-1)
                        elif model_type == 'cosine':
                            if warm_start == 'warm':
                                iv = torch.from_numpy(init_vec)
                                iv = iv / iv.norm()
                                iv = iv.type(torch.float32)
                            elif warm_start == 'default':
                                iv = None
                            mod = LookupVec(Xt.shape[1], margin=0.1, learning_rate=0.0003, init_vec=iv)
                            fit_rank(mod=mod, X=Xt.astype('float32'), y=yt.astype('float'), batch_size=minibatch_size)
                            tvec = mod.vec.detach().numpy().reshape(1,-1)
                        else:
                            assert False, 'model type'

                    else:
                        assert False
                else:
                    pass
                
        return res

def genruns(categories, qdict):
    tups = []
    for category in categories:
        initial_tup = {'category':category, 'qstr':qdict.get(category,category)}
    
        for variant,tup in [
            # ('random', dict(warm_start='default', interactive='plain', qstr='nolang')), #random all the time
            # ('nolang_interactive', dict(warm_start='default', interactive='pytorch', qstr='nolang')), #random init, then learn
            # ('nolang_interactive_sklearn', dict(warm_start='default', interactive='sklearn', qstr='nolang')), #random init, then learn
            # ('interactive_sklearn', dict(warm_start='default',interactive='sklearn')),
            ('interactive_pytorch_warm', dict(warm_start='warm', interactive='pytorch')),
            ('interactive_pytorch_cold', dict(warm_start='default',interactive='pytorch')),
            ('plain', dict(warm_start='default', interactive='plain')),
            ]:
            rtup = {'variant':variant}
            rtup.update(initial_tup)
            rtup.update(tup)
            tups.append(rtup)
    return tups

## would be great to have ray running on supercloud and just submit things from here... how is it different from 
## having a second machine?
## average precision @ 200
def make_prdf(dat, gt):
    hits = np.concatenate(dat['results'])
    nfound = hits.cumsum()
    ntotal = gt.sum()
    nframes = np.arange(hits.shape[0]) + 1
    precision = nfound/nframes
    recall = nfound/ntotal
    total_prec = (precision*hits).cumsum() # see Evaluation of ranked retrieval results page 158
    average_precision = total_prec/ntotal
    prdf = pd.DataFrame({'precision':precision, 'recall':recall, 'nfound':nfound,
                         'total_precision':total_prec,
                         'average_precision':average_precision,
                         'total':ntotal, 'nframes':nframes})
    return prdf


import math
def brief_format(ftpt):
    if math.isclose(ftpt, 0.):
        return '0'
    
    if math.isclose(ftpt,1.):
        return '1'

    if ftpt < 1.:
        exp = -math.floor(math.log10(abs(ftpt)))
        fmt_string = '{:.0%df}' % (exp + 1)
        dec = fmt_string.format(ftpt)    
    else:
        fmt_string = '{:.02f}'
        dec = fmt_string.format(ftpt)
        
    zstripped = dec.lstrip('0').rstrip('0')
    return zstripped.rstrip('.')


def times_format(ftpt):
    return brief_format(ftpt) + 'x'

def make_labeler(fmt_func):
    def fun(arrlike):
        return list(map(fmt_func, arrlike))
    return fun


import inspect
from .dataset_tools import *
from .fine_grained_embedding import *

def run_loop6(*, ev :EvDataset, category, qstr, interactive, warm_start, n_batches, batch_size, minibatch_size, 
              learning_rate, max_examples, num_epochs, loss_margin, tqdm_disabled:bool, fine_grained:bool, model_type='logistic', **kwargs):  
        def adjust_vec(vec, Xt, yt, learning_rate, loss_margin, max_examples, minibatch_size):
            vec = torch.from_numpy(vec).type(torch.float32)
            mod = LookupVec(Xt.shape[1], margin=loss_margin, optimizer=torch.optim.SGD, learning_rate=learning_rate, init_vec=vec)
            fit_rank2(mod=mod, X=Xt.astype('float32'), y=yt.astype('float'), 
                    max_examples=max_examples, batch_size=minibatch_size,max_epochs=1)
            newvec = mod.vec.detach().numpy().reshape(1,-1)
            return newvec        
        # gvec = ev.fine_grained_embedding
        # gvec_meta = ev.fine_grained_meta
        ev0 = ev
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        allargs = {k:v for (k,v) in values.items() if k in args and k not in ['ev', 'gvec', 'gvec_meta']}        

        ev, class_idxs = get_class_ev(ev0, category, boxes=True)
        dfds =  DataFrameDataset(ev.box_data[ev.box_data.category == category], index_var='dbidx', max_idx=class_idxs.shape[0]-1)
        ds = TxDataset(dfds, tx=lambda tup : resize_to_grid(224)(im=None, boxes=tup)[1])

        if fine_grained:
            vec_meta = ev.fine_grained_meta
            vecs = ev.fine_grained_embedding
            hdb = FineEmbeddingDB(raw_dataset=ev.image_dataset, embedding=ev.embedding, 
                embedded_dataset=vecs, vector_meta=vec_meta)
        else:
            dbidxs = np.arange(len(ev)).astype('int')
            vec_meta = pd.DataFrame({'iis': np.zeros_like(dbidxs), 'jjs':np.zeros_like(dbidxs), 'dbidx':dbidxs})
            vecs = ev.embedded_dataset
            hdb = EmbeddingDB(raw_dataset=ev.image_dataset, embedding=ev.embedding,embedded_dataset=vecs)

        bfq = BoxFeedbackQuery(hdb, batch_size=batch_size, auto_fill_df=None)
        rarr = ev.query_ground_truth[category]
        print(allargs, kwargs)        
                
        if qstr == 'nolang':
            init_vec=None
            init_mode = 'random'
        else:
            init_vec = ev.embedding.from_string(string=qstr)
            init_mode = 'dot'
        
        acc_pos = []
        acc_neg = []
        acc_indices = []
        acc_results = []
        acc_ranks = []
        total = 0
                
        gt = ev.query_ground_truth[category].values
        res = {'indices':acc_indices, 'results':acc_results}#, 'gt':gt.values.copy()}
        if init_vec is not None:
            tvec = init_vec/np.linalg.norm(init_vec)
        else:
            tvec = None

        tmode = init_mode

        for i in tqdm(range(n_batches), leave=False, disable=tqdm_disabled):
            idxbatch = bfq.query_stateful(mode=tmode, vector=tvec, batch_size=batch_size)
            acc_indices.append(idxbatch)
            acc_results.append(gt[idxbatch])

            if interactive != 'plain':
                if fine_grained:
                    batchpos, batchneg = get_pos_negs_all(idxbatch, ds, vec_meta)
                    acc_pos.append(batchpos)
                    acc_neg.append(batchneg)

                    pos = pr.BitMap.union(*acc_pos)
                    neg = pr.BitMap.union(*acc_neg)
                    # print('pos, neg', len(pos), len(neg))
                    Xt = np.concatenate([vecs[pos], vecs[neg]])
                    yt = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

                    if np.concatenate(acc_results).sum() > 0:
                        assert len(pos) > 0
                else:
                    Xt = vecs[idxbatch]
                    yt = gt[idxbatch]

                if (yt.shape[0] > 0) and (yt.max() > yt.min()):
                    tmode = 'dot'
                    if interactive == 'sklearn':
                        lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
                        lr.fit(Xt, yt)
                        tvec = lr.coef_.reshape(1,-1)        
                    elif interactive == 'pytorch':
                        p = yt.sum()/yt.shape[0]
                        w = np.clip((1-p)/p, .1, 10.)

                        if model_type == 'logistic':
                            mod = PTLogisiticRegression(Xt.shape[1], learning_rate=learning_rate, C=0, 
                                                        positive_weight=w)
                            if warm_start == 'warm':
                                iv = torch.from_numpy(init_vec)
                                iv = iv / iv.norm()
                                mod.linear.weight.data = iv.type(mod.linear.weight.dtype)
                            elif warm_start == 'default':
                                pass

                            fit_reg(mod=mod, X=Xt.astype('float32'), y=yt.astype('float'), batch_size=minibatch_size)
                            tvec = mod.linear.weight.detach().numpy().reshape(1,-1)
                        elif model_type in ['cosine', 'multirank']:
                            for i in range(num_epochs):
                                tvec = adjust_vec(tvec, Xt, yt, learning_rate=learning_rate, 
                                                  max_examples=max_examples, 
                                                  minibatch_size=minibatch_size,
                                                  loss_margin=loss_margin)
                        else:
                            assert False, 'model type'

                    else:
                        assert False
                else:
                    # print('missing positives or negatives to do any training', yt.shape, yt.max(), yt.min())
                    pass
                
        res['indices'] = [class_idxs[r] for r in res['indices']]
        tup = {**allargs, **kwargs}
        return (tup, res)
