import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class TempRegression(pl.LightningModule):
    def __init__(self, in_features, out_classes, initial_weights=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_classes)
        self.log_temps = nn.Parameter(
            data=torch.zeros(out_classes, dtype=self.linear.weight.dtype)
        )

    def forward(self, X):
        normalized_weight = F.normalize(self.linear.weight, dim=-1)
        return (X @ normalized_weight.t()) * self.log_temps.reshape(
            1, -1
        ).exp() + self.linear.bias

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW([self.linear.weight], lr=0.001, weight_decay=0.0)
        opt.add_param_group(
            {"params": [self.linear.bias], "weight_decay": 0.0, "lr": 0.05}
        )
        opt.add_param_group(
            {"params": [self.log_temps], "weight_decay": 0.0, "lr": 0.005}
        )

        return opt


class RegressionBase(pl.LightningModule):
    def __init__(self, in_features, out_classes, initial_weights=None):
        super().__init__()
        self.out_classes = out_classes
        self.linear = nn.Linear(in_features, out_classes)

    @torch.no_grad()
    def get_vec(self):
        return self.linear.weight[0, :].numpy()

    def forward(self, X, y=None):
        score = self.linear(X)
        if y is not None:
            if self.out_classes == 1:
                loss = F.binary_cross_entropy_with_logits(score, y)
            else:
                loss = F.cross_entropy(score, y)
            return {"loss": loss}
        else:
            if self.out_classes == 1:
                return F.sigmoid(score)
            else:
                return F.softmax(score, dim=-1)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        ret = self(x, y)
        loss = ret["loss"]
        # self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        ret = self(x, y)
        loss = ret["loss"]
        # self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW([self.linear.weight], lr=0.001, weight_decay=0.0)
        opt.add_param_group(
            {"params": [self.linear.bias], "weight_decay": 0.0, "lr": 0.01}
        )

        return opt


## want to try: softmax model vs binary model online
## want to try: separate lookup and rescoring
## want to try : different model optimization strategies (init vs regularized)
import pyroaring as pr
from sklearn.metrics import average_precision_score
from scipy.special import log_softmax


class SimpleLoop:
    def __init__(self, objmx, train_df, val_df, cat, k):
        self.objmx = objmx
        self.qvec0 = objmx.string2vec(cat).reshape(-1)
        self.weight_map = {cat: self.qvec0}
        self.k = k
        self.cat = cat

        self.train_df = train_df
        self.dbvecs = train_df.vectors.to_numpy()
        self.dblabels = train_df.category

        self.val_df = val_df

        self.val_xs = val_df.vectors.to_numpy()
        self.val_ys = val_df.category == cat
        self.rounds = 0

    def top_k(self, qvec, k):
        scores = -(self.dbvecs @ qvec)
        topk = np.argsort(scores)[:k]
        xvecs = self.dbvecs[topk]
        labs = self.dblabels.iloc[topk].values.to_numpy()
        return xvecs, labs, scores

    def eval_vec(self, qvec):
        score = self.dbvecs @ qvec
        return average_precision_score(self.val_ys, score)

    @torch.no_grad()
    def eval_mod(self, mod):
        pred_mcs = mod(torch.from_numpy(self.val_xs)).numpy()
        val_scores = pred_mcs[:, 0]
        sc1 = average_precision_score(self.val_ys, val_scores)

        vec = mod.get_vec()
        unormalized_scores = self.val_xs @ vec
        sc2 = average_precision_score(self.val_ys, unormalized_scores)

        print("val ap", sc2, "normalized ", sc1)

    def make_mod(self, labels):
        ## add init new vec for new labels
        for lab in labels:
            if lab not in self.weight_map:
                self.weight_map[lab] = self.objmx.string2vec(lab).reshape(-1)

        mod = RegressionBase(in_features=512, out_classes=labels.shape[0])
        initial_weights = torch.from_numpy(
            np.stack([self.weight_map[k] for k in labels])
        )
        mod.linear.load_state_dict({"weight": initial_weights.clone()}, strict=False)
        return mod

    def closure(self, niter):
        cat = self.cat
        weight_map = self.weight_map
        qvec = self.weight_map[self.cat]
        xvecs, labs, _ = self.top_k(qvec, k=(self.rounds + 1) * self.k)

        unique_labels = np.unique(labs)
        labels = np.array([self.cat] + [c for c in unique_labels if c != cat])
        label_map = {lab: k for k, lab in enumerate(labels)}

        mod = self.make_mod(labels)
        self.eval_mod(mod)

        opt = mod.configure_optimizers()

        X = torch.from_numpy(xvecs)
        y = torch.tensor([label_map[c] for c in labs], dtype=torch.long)

        for i in range(niter):
            loss = mod.training_step((X, y), None)
            print("  training loss: ", loss.item())
            loss.backward()
            opt.step()

        with torch.no_grad():
            for (i, vec) in enumerate(mod.linear.weight):
                l = labels[i]
                weight_map[l] = vec.detach().numpy()

        self.eval_mod(mod)
        self.rounds += 1


#     for i in range(10):
#         closure((i+1)*nk, niter=3)
