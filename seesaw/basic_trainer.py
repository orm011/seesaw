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
        def closure():
            self.opt.zero_grad()
            ret = self.mod.training_step(batch, batch_idx)
            
            weight = self.opt._params[0]
            prev_grad = torch.zeros_like(weight)

            if self.verbose:
                for (k,v) in ret.items():
                    if k == 'loss':
                        continue
                    
                    v.backward(retain_graph=True)
                    grad_change = (weight.grad - prev_grad)
                    if self.verbose:
                        print(f'{k=} {v.detach().item()=} {grad_change.norm()=}')
                    prev_grad = weight.grad.clone()

                self.opt.zero_grad()

            ret['loss'].backward()

            if self.verbose:
                ## check the grad from calling one by one is the same as the total grad accumulated?
                pass
                
            return ret['loss']

        for _ in range(self.max_epochs):
            for batch_idx, batch in enumerate(train_loader):
                self.opt.step(closure)

        return None

    def validate(self, dataloader):
        self.mod.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                ret = self.mod.validation_step(batch, batch_idx)

        return ret



