import torch.utils.data

class HCatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, xforms=lambda x: x):
        self.datasets = tuple(datasets)
        self.__len = min([len(x) for x in self.datasets])
        self.xforms = xforms

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        ans = [ds[idx] for ds in self.datasets]
        return self.xforms(ans)

class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df, index_var, max_idx=None, xforms=None):
        self.df = df
        self.xforms = (lambda x: x) if xforms is None else xforms
        self.max_idx = (df[index_var].max()) if (max_idx is None) else max_idx
        self.index_var = index_var

    def __len__(self):
        return self.max_idx + 1

    def __getitem__(self, idx):
        assert idx <= self.max_idx
        quals = self.df[self.index_var] == idx
        return self.xforms(self.df[quals])