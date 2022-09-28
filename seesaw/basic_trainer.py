import pytorch_lightning as pl
import torch

class SimpleTrainer:
    def __init__(self, mod : pl.LightningModule, max_epochs):
        self.mod = mod
        self.max_epochs = max_epochs
        self.opt = mod.configure_optimizers()

    def fit(self, train_loader):
        self.mod.train()
        for _ in range(self.max_epochs):
            for batch_idx, batch in enumerate(train_loader):
                self.opt.zero_grad()
                ret = self.mod.training_step(batch, batch_idx)
                loss = ret['loss']
                loss.backward()
                self.opt.step()

    def validate(self, dataloader):
        self.mod.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                ret = self.mod.validation_step(batch, batch_idx)

        return ret



