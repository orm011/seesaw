import torch.utils.data

class HCatDataset(object):
    def __init__(self, datasets, xforms=lambda x: x):
        self.datasets = tuple(datasets)
        self.__len = min([len(x) for x in self.datasets])
        self.xforms = xforms

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        ans = [ds[idx] for ds in self.datasets]
        return self.xforms(ans)

class DataFrameDataset(object):
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


class TxDataset(object):
    def __init__(self, ds, tx):
        self.ds = ds
        self.tx = tx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.tx(self.ds[idx])

class ExplicitPathDataset(object):
    def __init__(self, root_dir, relative_path_list):
        '''
        Reads images in a directory according to an explicit list.
        '''
        self.root = root_dir
        self.paths = relative_path_list

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        relpath = self.paths[idx].lstrip('./')
        image = PIL.Image.open('{}/{}'.format(self.root, relpath))
        return {'file_path':relpath, 'dbidx':idx, 'image':image}