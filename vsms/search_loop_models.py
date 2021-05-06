
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CalibrationLayer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        learning_rate: float = 1e-3,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        l2_strength: float = 0.0,
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
        self.linear = nn.Sequential(
                        nn.Linear(in_features=self.hparams.input_dim, 
                                out_features=self.hparams.hidden_dim, 
                                bias=bias),
                        nn.Tanh(),
                        nn.Linear(in_features=self.hparams.hidden_dim,
                                  out_features=self.hparams.hidden_dim,
                                  bias=bias),
                        # nn.Tanh(),
                        # nn.Linear(in_features=self.hparams.hidden_dim,
                        #           out_features=self.hparams.input_dim,
                        #           bias=bias)
                                  )
        self.scaling = nn.Parameter(data=torch.tensor([0.]))
        # self.log_temperature = nn.Parameter(data=torch.tensor([0.]))        
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.zeros_(m.weight)#, a=-1./self.hparams., b=1./m.weight.shape[1])
                torch.nn.init.zeros_(m.bias)
                
        self.linear.apply(init_weights)
            
    def forward(self, qvec):
        qvec = self.linear(qvec) + qvec
        return qvec

    def _batch_step(self, batch, batch_idx):
        str_vecs, im_vecs = batch
        str_vecs = self(str_vecs)
        
        # normalize features
        str_vecs = str_vecs / str_vecs.norm(dim=-1, keepdim=True)
        #im_vecs = im_vecs
        #  temp = self.log_temperature.exp()
        #    logits_im = self.scaling.exp()*
        logits_im = (im_vecs @ str_vecs.t())
        logits_cap = logits_im.t()#self.scaling.exp()*(im_vecs @ str_vecs.t())
        
        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss1 = F.cross_entropy(logits_im, torch.arange(logits_im.shape[0]), reduction='sum')
        loss2 = 0#F.cross_entropy(logits_cap, torch.arange(logits_im.shape[0]), reduction='sum')
        loss = loss1 + loss2

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg

        loss /= str_vecs.size(0)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._batch_step(batch, batch_idx)
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self._batch_step(batch, batch_idx)
        self.log('loss/val', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

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
from torch.utils.data import DataLoader

def fit(mod, X, y, valX=None, valy=None, logger=None, batch_size=9, max_epochs=10, gpus=1, precision=16):
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    
    train_ds = TensorDataset(X,torch.from_numpy(y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    if valX is not None:
        if not torch.is_tensor(valX):
            valX = torch.from_numpy(valX)
        val_ds = TensorDataset(valX, torch.from_numpy(valy))
    else:
        val_ds = train_ds
    
    val_loader = DataLoader(val_ds, batch_size=1000, shuffle=False, num_workers=0)

    if logger is None:
        logger = pl.loggers.TensorBoardLogger(save_dir='fit_method')
    
    trainer = pl.Trainer(logger=logger, gpus=gpus, precision=precision, max_epochs=max_epochs,
                         callbacks=[pl.callbacks.early_stopping.EarlyStopping(monitor='loss/val', patience=2)], 
                         progress_bar_refresh_rate=1)
    #  mod = models_ptlr_c0_w40[c]
    trainer.fit(mod, train_loader, val_loader)