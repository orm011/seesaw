from .dbserver import *

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


def run_loop(*, dbactor, category, qstr, interactive, warm_start, n_batches, batch_size, **kwargs):
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
                        lr = PTLogisiticRegression(Xt.shape[1], learning_rate=.0003, C=0, positive_weight=w)

                        if warm_start == 'warm':
                            iv = torch.from_numpy(init_vec)
                            iv = iv / iv.norm()
                            lr.linear.weight.data = iv.type(lr.linear.weight.dtype)
                        elif warm_start == 'default':
                            pass
                        else:
                            assert False, 'assert on warm start'

                        fit(mod=lr, X=Xt.astype('float32'), y=yt.astype('float'), batch_size=batch_size)
                        tvec = lr.linear.weight.detach().numpy().reshape(1,-1)                        
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


def brief_format(ftpt):
    if ftpt == 0.:
        return '0'
    
    if ftpt == 1.:
        return '1'
    
    if .01 < ftpt < 1.:
        dec = '{:.02f}'.format(ftpt)
    else:
        dec = '{:.05f}'.format(ftpt)
    
    zstripped = dec.lstrip('0').rstrip('0')
    return zstripped.rstrip('.')

def times_format(ftpt):
    return brief_format(ftpt) + 'x'

def make_labeler(fmt_func):
    def fun(arrlike):
        return list(map(fmt_func, arrlike))
    return fun
