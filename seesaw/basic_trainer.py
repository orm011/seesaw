import torch

class BasicTrainer:
    ## mod has an interface like lightning_module
    def __init__(self, mod , max_epochs):
        self.mod = mod
        self.max_epochs = max_epochs
        self.opt = mod.configure_optimizers()

    def fit(self, train_loader):
        self.mod.train()
        rets = []
        def closure():
            self.opt.zero_grad()
            ret = self.mod.training_step(batch, batch_idx)
            loss = ret['loss']
            lossval = loss.detach().item()
            ret['loss'] = lossval
            rets.append(ret)
            loss.backward()
            return loss

        for _ in range(self.max_epochs):
            for batch_idx, batch in enumerate(train_loader):
                self.opt.step(closure)

        return rets

    def validate(self, dataloader):
        self.mod.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                ret = self.mod.validation_step(batch, batch_idx)

        return ret



