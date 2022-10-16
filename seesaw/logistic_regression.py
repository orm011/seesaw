
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from .basic_trainer import BasicTrainer

import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class LogisticRegModule(nn.Module):
    def __init__(self, *, dim,  pos_weight=1., reg_weight=1., fit_intercept=True,  verbose=False, max_iter=100, lr=1., regularizer_function=None):
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=fit_intercept)
        self.pos_weight = torch.tensor([pos_weight])
        self.regularizer_function = regularizer_function
            
        self.reg_weight = reg_weight
        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose
        
    def get_coeff(self):
        return self.linear.weight.detach().numpy()

    def forward(self, X, y=None):
        logits =  self.linear(X)
        if y is None:
            return logits.sigmoid()
        else:
            return logits
    
    def _step(self, batch):
        X,y=batch # note y can be a floating point
        logits = self(X, y)
        weighted_celoss = F.binary_cross_entropy_with_logits(logits, y, 
                                    reduction='none', pos_weight=self.pos_weight)     
        return weighted_celoss
        
    
    def training_step(self, batch, batch_idx):
        celoss = self._step(batch).mean()
        if self.regularizer_function is None:
            reg = self.linear.weight.norm()
        else:
            reg = self.regularizer_function()
        
        loss = celoss + self.reg_weight*reg
        if self.verbose:
            print('wnorm', self.linear.weight.norm().detach().item())
            if self.linear.bias is not None:
                print('bias', self.linear.bias.detach().item())
        
        return {'loss':loss, 'celoss':celoss, 'reg':reg}
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {'loss':loss.mean()}

    def configure_optimizers(self):
        return opt.LBFGS(self.parameters(), max_iter=self.max_iter, lr=self.lr, line_search_fn='strong_wolfe')

import numpy as np
from sklearn.decomposition import PCA

class LogisticRegresionPT: 
    def __init__(self, class_weights, scale='centered',  reg_lambda=1., verbose=False, 
            regularizer_vector=None,  fit_intercept=False, **kwargs):
        ''' reg type: nparray means use that vector '''
        assert scale in ['centered', None]
        self.class_weights = class_weights
        self.kwargs = kwargs
        self.model_ = None
        self.trainer_ = None
        self.mu_ = None
        self.scale = scale
        self.reg_lambda = reg_lambda
        self.n_examples = None
        self.regularization_type = None
        self.regularizer_vector = None
        self.verbose = verbose
        self.fit_intercept = fit_intercept

        if isinstance(regularizer_vector, np.ndarray):
            self.regularizer_vector = F.normalize(torch.from_numpy(regularizer_vector.reshape(1,-1)).float(), dim=-1).reshape(-1)
            self.regularization_type = 'vector'
        elif isinstance(regularizer_vector, str):
            self.regularization_type = regularizer_vector
        elif regularizer_vector is None:
            self.regularization_type = None
        else:
            assert False
            

        if scale == 'centered':
            self.scaler_ = StandardScaler(with_mean=True, with_std=False)
        else:
            self.scaler_ = None

    def _regularizer_func(self):
        assert self.model_ is not None

        if self.regularization_type is None:
            return 0.
        
        weight = self.model_.linear.weight

        if self.regularizer_vector is not None:
            if self.scale == 'centered' or self.scale is None :
                base_vec = self.regularizer_vector
            else:
                assert False

            norm_penalty = (weight.norm() - 1.)**2
            angle_penalty = (F.normalize(weight).reshape(-1) - base_vec.reshape(-1)).norm()**2
        else:
            if self.regularization_type in ['norm1', 'norm']:
                norm_target = 1 if self.regularization_type == 'norm1' else 0
                norm_penalty = (weight.norm() - norm_target)**2
            else:
                assert False
                
            angle_penalty = 0.

        ans = norm_penalty + angle_penalty
        return ans 
            
    def _get_coeff(self):
        assert self.model_
        weight_prime = self.model_.linear.weight
        if self.scale == 'centered':
            return weight_prime
        else:
            assert False

    def get_coeff(self):
        return self._get_coeff().detach().numpy()

    def _get_intercept(self):
        assert self.model_
        return -self._get_coeff()@self.mu_.reshape(-1) + self.model_.linear.bias

    def get_intercept(self):
        return self._get_intercept().detach().numpy()
    
    def fit(self, X, y):
        self.n_examples = X.shape[0]
        npos = (y == 1).sum()
        nneg = (y == 0).sum() 

        if self.scaler_:
            X = self.scaler_.fit_transform(X)
            self.mu_ = torch.from_numpy(self.scaler_.mean_).float()

        if self.class_weights == 'balanced':
                pseudo_pos = max(npos, 1)
                pseudo_neg = max(nneg, 1)
                pos_weight = pseudo_neg / pseudo_pos
        else:
            pos_weight = self.class_weights
        
        self.model_ = LogisticRegModule(dim=X.shape[1], pos_weight=pos_weight, 
                        reg_weight=self.reg_lambda/self.n_examples,
                        fit_intercept=self.fit_intercept, #(npos > 0 and nneg > 0), # only fit intercept if there are both signs
                        regularizer_function=self._regularizer_func, **self.kwargs)
        
        if self.regularizer_vector is None:
            self.regularizer_vector = torch.zeros_like(self.model_.linear.weight)

        ds = TensorDataset(torch.from_numpy(X),torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=len(ds), shuffle=True)
        self.trainer_ = BasicTrainer(mod=self.model_, max_epochs=1)
        self.losses_ = self.trainer_.fit(dl)

        is_nan = False        
        for i,loss in enumerate(self.losses_):
            if math.isnan(loss) or math.isinf(loss):
                if self.verbose:
                    print(f'warning: loss diverged at step {i=} {loss=:.03f}. you may want to consider scaling and centering')
                is_nan = True
                break
        
        if not is_nan:
            niter = len(self.losses_)
            if self.verbose:
                print(f'converged after {niter} iterations. {self.losses_[-1]=}')
                
    def predict_proba(self, X):
        if self.scaler_:
            X = self.scaler_.transform(X)
        
        with torch.no_grad():
            return self.model_(torch.from_numpy(X)).numpy()

    def predict_proba2(self, X): # test
        X = torch.from_numpy(X)
        with torch.no_grad():
            logits =  X @ self._get_coeff().reshape(-1)   + self._get_intercept()
            ps = logits.sigmoid()

        return ps.reshape(-1,1).numpy()
