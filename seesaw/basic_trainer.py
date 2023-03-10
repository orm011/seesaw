import torch

class BasicTrainer:
    ## mod has an interface like lightning_module
    def __init__(self, mod , max_epochs, verbose=False):
        self.mod = mod
        self.max_epochs = max_epochs
        self.opt = mod.configure_optimizers()
        self.verbose = verbose

    def fit(self, train_loader):
        self.mod.train()

        batch = (None,None)
        batch_idx = None

        loss_history = []
        def closure():
            self.opt.zero_grad()
            ret = self.mod.training_step(batch, batch_idx)            
            weight = self.opt._params[0]
            prev_grad = torch.zeros_like(weight)

            if self.verbose:
                entry = []
                for (k,v) in ret.items():
                    if k == 'loss':
                        continue
                    
                    v.backward(retain_graph=True)
                    grad_change = (weight.grad - prev_grad)
                    entry.append({'k':k, 'loss':v.detach().item(), 'grad_norm':grad_change.norm().item()})
                    prev_grad = weight.grad.clone()

                loss_history.extend(entry)
                self.opt.zero_grad()

            ret['loss'].backward()

            if self.verbose:
                assert torch.isclose(weight.grad, prev_grad, atol=1e-5).all(), f'{weight.grad.norm()=} {prev_grad.norm()=}'

            return ret['loss']

        for _ in range(self.max_epochs):
            if train_loader is not None:
                for batch_idx, batch in enumerate(train_loader):
                    self.opt.step(closure)
            else:
                self.opt.step(closure)

        return loss_history

    def validate(self, dataloader):
        self.mod.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                ret = self.mod.validation_step(batch, batch_idx)

        return ret



