
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Subset
from tqdm.auto import tqdm

class PTLogisiticRegression(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        bias: bool = True,
        learning_rate: float = 5e-3,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        C: float = 0.0,
        positive_weight : float = 1.0,
        **kwargs
    ):
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='Adam')
            l1_strength: L1 regularization strength (default=None)
            l2_strength: L2 regularization strength (default=None)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.linear = nn.Linear(in_features=self.hparams.input_dim, 
                                out_features=1,
                                bias=bias)
        
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hparams.positive_weight]).float(), reduction='none')
        self.average_precision = pl.metrics.AveragePrecision(num_classes=2, pos_label=1, 
                                                               compute_on_step=False)

    def forward(self, qvec):
        qvec = self.linear(qvec)
        return qvec
    
    def _batch_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.loss(logits, y.view(-1,1)).reshape(-1).mean()
        
        # L2 regularizer
        if self.hparams.C > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss = self.hparams.C*loss + l2_reg

        loss /= y.size(0)
        return {'loss':loss, 'logits':logits, 'y':y}

    def training_step(self, batch, batch_idx):
        d = self._batch_step(batch, batch_idx)
        loss = d['loss']
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        d = self._batch_step(batch, batch_idx)
        loss = d['loss']
        logits = torch.clone(d['logits']).view(-1)
        
        self.average_precision(logits, d['y'].view(-1).long())
        self.log('loss/val', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'logits':logits, 'y':d['y']}
    
    def validation_epoch_end(self, validation_step_outputs):
        apval = self.average_precision.compute()
        self.log('AP/val', apval, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.average_precision.reset()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

from torch.utils.data import TensorDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader

# def fit(mod, X, y, valX=None, valy=None, logger=None, batch_size=9, max_epochs=10, gpus=1, precision=16):
#     if not torch.is_tensor(X):
#         X = torch.from_numpy(X)
    
#     train_ds = TensorDataset(X,torch.from_numpy(y))
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

#     if valX is not None:
#         if not torch.is_tensor(valX):
#             valX = torch.from_numpy(valX)
#         val_ds = TensorDataset(valX, torch.from_numpy(valy))
#     else:
#         val_ds = train_ds
    
#     val_loader = DataLoader(val_ds, batch_size=1000, shuffle=False, num_workers=0)

#     if logger is None:
#         logger = pl.loggers.TensorBoardLogger(save_dir='fit_method')
    
#     trainer = pl.Trainer(logger=logger, gpus=gpus, precision=precision, max_epochs=max_epochs,
#                          callbacks=[pl.callbacks.early_stopping.EarlyStopping(monitor='loss/val', patience=2)], 
#                          progress_bar_refresh_rate=1)
#     #  mod = models_ptlr_c0_w40[c]
#     trainer.fit(mod, train_loader, val_loader)

class LookupVec(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        margin: float = .1,
        init_vec: torch.tensor = None,
        learning_rate: float = 5e-3,
        positive_weight: float = 1.,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        **kwargs
    ):
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
        """
        print(optimizer)
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        
        if init_vec is not None:
            self.vec = nn.Parameter(init_vec.reshape(1,-1))
        else:
            t = torch.randn(1,input_dim)
            self.vec = nn.Parameter(t/t.norm())

        # self.loss = nn.CosineEmbeddingLoss(margin,reduction='none')
        self.rank_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, qvec):
        return F.cosine_similarity(self.vec, qvec)
    
    def _batch_step(self, batch, batch_idx):
        X1, X2, y = batch
        sim1 = self(X1)
        sim2 = self(X2)
        
        losses = self.rank_loss(sim1, sim2, y.view(-1,1)).reshape(-1)
        return {'loss':losses.mean(),  'y':y}

    def training_step(self, batch, batch_idx):
        d = self._batch_step(batch, batch_idx)
        loss = d['loss']
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        d = self._batch_step(batch, batch_idx)
        loss = d['loss']
        
        self.log('loss/val', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'y':d['y']}
    
    def validation_epoch_end(self, validation_step_outputs):
        apval = self.average_precision.compute()
        self.log('AP/val', apval, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.average_precision.reset()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

def fit_rank(*, mod, X, y, batch_size, valX=None, valy=None, logger=None,  max_epochs=4, gpus=0, precision=32):
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
    
    def make_tuple_ds(X, y, max_size):
        X1ls = []
        X2ls = []
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if y[i] > y[j]:
                    X1ls.append(X[i])
                    X2ls.append(X[j])

        X1 = torch.stack(X1ls)
        X2 = torch.stack(X2ls)
        train_ds = TensorDataset(X1,X2, torch.ones(X1.shape[0]))
        if len(train_ds) > max_size:
            ## ranom sample
            randsel = torch.randperm(len(train_ds))[:max_size]
            train_ds = Subset(train_ds, randsel)
        return train_ds

    train_ds = make_tuple_ds(X, y, max_size=4*X.shape[0])
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
