from seesaw.dataset_manager  import GlobalDataManager
from seesaw.research.knn_methods import KNNGraph, uniquify_knn_graph
from seesaw.seesaw_session import get_subset
import os

class IndexActor:
    def __init__(self, root):
        gdm = GlobalDataManager(root)
        ds = gdm.get_dataset('lvis')
        idx_top = ds.load_index('multiscale',  options=dict(use_vec_index=False))
        knng = KNNGraph.from_file(f'/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/data/lvis/indices/multiscale/subsets/{category}/')
        self.gdm = gdm
        self.ds = ds
        self.idx_top = idx_top
        self.knng = knng
        
    def process_category(self, category):
        idx, _, _, _ = get_subset(self.ds, self.idx_top, c_name=category)
        ppdf = uniquify_knn_graph(self.knng, idx)
        odir = f'/home/gridsan/omoll/fastai_shared/omoll/seesaw_root2/data/lvis/indices/multiscale/subsets/{category}/dividx/'
        os.makedirs(odir, exist_ok=True)
        ppdf.to_parquet(f'{odir}/sym.parquet')


