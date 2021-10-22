import clip
import pytorch_lightning as pl
import os
import multiprocessing as mp

import clip, torch
import PIL
from operator import itemgetter

from .multigrain import * #SlidingWindow,PyramidTx,non_resized_transform
from .embeddings import * 
from .vloop_dataset_loaders import EvDataset

import glob
import pandas as pd
from tqdm.auto import tqdm
 
import torch
import torch.nn as nn
import pyroaring as pr
from torch.utils.data import Subset
import ray
import shutil


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


def list_image_paths(basedir, extensions=['jpg', 'jpeg', 'png']):
    acc = []
    for ext in extensions:
        imgs = glob.glob(f'{basedir}/**/*.{ext}', recursive=True)
        acc.extend(imgs)
        
    relative_paths = [ f[len(basedir):].lstrip('./') for f in acc]
    return relative_paths

import math


def preprocess(tup):
    """ meant to preprocess dict with {path, dbidx,image}
    """
    ptx = PyramidTx(tx=non_resized_transform(224), factor=math.sqrt(.5), min_size=224)
    ims, sfs = ptx(tup['image'])
    acc = []
    for zoom_level,(im,sf) in enumerate(zip(ims,sfs), start=1):
        acc.append({'file_path':tup['file_path'],'dbidx':tup['dbidx'], 
                    'image':im, 'scale_factor':sf, 'zoom_level':zoom_level})
        
    return acc

class NormalizedEmbedding(nn.Module):
    def __init__(self, emb_mod):
        super().__init__()
        self.mod = emb_mod

    def forward(self, X):
        tmp = self.mod(X)
        with torch.cuda.amp.autocast():
            return F.normalize(tmp, dim=1).type(tmp.dtype)


def trace_emb_jit(output_path):
    device = torch.device('cuda:0')
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    ker = NormalizedEmbedding(model.visual)
    ker = ker.eval()

    example = torch.randn((10,3,224,224), dtype=torch.half, device=torch.device('cuda:0'))
    with torch.no_grad():
        jitmod = torch.jit.trace(ker, example)

    out = ker(example)
    print(out.dtype)
    jitmod.save(output_path)

class ImageEmbedding(nn.Module):
    def __init__(self, device, jit_path=None):
        super().__init__()
        self.device = device            

        if jit_path == None:
            model, _ = clip.load("ViT-B/32", device=device, jit=False)
            ker = NormalizedEmbedding(model.visual)
        else:
            ker = torch.jit.load(jit_path, map_location=device)

        
        kernel_size = 224 # changes with variant        
        self.model = SlidingWindow(ker, kernel_size=kernel_size, stride=kernel_size//2,center=True)

    def forward(self, *, preprocessed_image):
        return self.model(preprocessed_image)
    
def postprocess_results(acc):
    flat_acc = {'iis':[], 'jjs':[], 'dbidx':[], 'vecs':[], 'zoom_factor':[], 'zoom_level':[],
               'file_path':[]}
    flat_vecs = []

    #{'accs':accs, 'sf':sf, 'dbidx':dbidx, 'zoom_level':zoom_level}
    for item in acc:
        acc0,sf,dbidx,zl,fp = itemgetter('accs', 'scale_factor', 'dbidx', 'zoom_level', 'file_path')(item)
        acc0 = acc0.squeeze(0)
        acc0 = acc0.transpose((1,2,0))

        iis, jjs = np.meshgrid(np.arange(acc0.shape[0], dtype=np.int16), np.arange(acc0.shape[1], dtype=np.int16), indexing='ij')
        #iis = iis.reshape(-1, acc0)
        iis = iis.reshape(-1)
        jjs = jjs.reshape(-1)
        acc0 = acc0.reshape(-1,acc0.shape[-1])
        imids = np.ones_like(iis)*dbidx
        zf = np.ones_like(iis)*(1./sf)
        zl = np.ones_like(iis)*zl

        flat_acc['iis'].append(iis)
        flat_acc['jjs'].append(jjs)
        flat_acc['dbidx'].append(imids)
        flat_acc['vecs'].append(acc0)
        flat_acc['zoom_factor'].append(zf.astype('float32'))
        flat_acc['zoom_level'].append(zl.astype('int16'))
        flat_acc['file_path'].append([fp]*iis.shape[0])

    flat = {}
    for k,v in flat_acc.items():
        flat[k] = np.concatenate(v)

    vecs = flat['vecs']
    del flat['vecs']

    vec_meta = pd.DataFrame(flat)
    # vecs = vecs.astype('float32')
    # vecs = vecs/(np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-6)
    vec_meta = vec_meta.assign(**get_boxes(vec_meta), vectors=TensorArray(vecs))
    return vec_meta.drop(['iis', 'jjs'],axis=1)
    
class BatchInferModel:
    def __init__(self, model, device):
        self.device = device
        self.model = model 
        
    def __call__(self, batch):
        with torch.no_grad():
            res = []
            for tup in batch:
                im = tup['image']
                del tup['image']
                vectors = self.model(preprocessed_image=im.unsqueeze(0).to(self.device)).to('cpu').numpy()
                tup['accs'] = vectors
                res.append(tup)
        return postprocess_results(res)

#mcc = mini_coco(None)
def worker_function(wid, dataset,slice_size,indices, cpus_per_proc, vector_root, jit_path):
    wslice = indices[wid*slice_size:(wid+1)*slice_size]
    dataset = Subset(dataset, indices=wslice)
    device = f'cuda:{wid}'
    extract_seesaw_meta(dataset, output_dir=vector_root, output_name=f'part_{wid:03d}', num_workers=cpus_per_proc, 
                    batch_size=3,device=device, jit_path=jit_path)
    print(f'Worker {wid} finished.')


def iden(x):
    return x


class Preprocessor:
    def __init__(self, jit_path, output_dir, meta_dict):
        emb = ImageEmbedding(device='cuda:0', jit_path=jit_path)
        self.bim = BatchInferModel(emb, 'cuda:0')
        self.output_dir = output_dir
        self.num_cpus = int(os.environ.get('OMP_NUM_THREADS'))
        self.meta_dict = meta_dict

    #def extract_meta(self, dataset, indices):
    def extract_meta(self, ray_dataset):
        # dataset = Subset(dataset, indices=indices)
        # txds = TxDataset(dataset, tx=preprocess)

        meta_dict = self.meta_dict

        def fix_meta(ray_tup):
            fullpath,binary = ray_tup
            p = os.path.realpath(fullpath)
            file_path,dbidx = meta_dict[p]
            return {'file_path':file_path, 'dbidx':dbidx, 'binary':binary}

        def full_preproc(tup):
            ray_tup = fix_meta(tup)
            image = PIL.Image.open(io.BytesIO(ray_tup['binary']))
            ray_tup['image'] = image
            del ray_tup['binary']
            return preprocess(ray_tup)

        def preproc_batch(b):
            return [full_preproc(tup) for tup in b]

        dl = ray_dataset.pipeline(parallelism=20).map_batches(preproc_batch, batch_size=20)
        res = []
        for batch in dl.iter_rows():
            batch_res = self.bim(batch)
            res.append(batch_res)
        # dl = DataLoader(txds, num_workers=1, shuffle=False,
        #                 batch_size=1, collate_fn=iden)
        # res = []
        # for batch in dl:
        #     flat_batch = sum(batch,[])
        #     batch_res = self.bim(flat_batch)
        #     res.append(batch_res)

        merged_res = pd.concat(res, ignore_index=True)
        ofile = f'{self.output_dir}/task_{ray.get_runtime_context().task_id}.parquet'

        ### TMP: parquet does not allow half prec.
        x = merged_res
        x = x.assign(vectors=TensorArray(x['vectors'].to_numpy().astype('single')))
        x.to_parquet(ofile)
        return ofile

def extract_seesaw_meta(dataset, output_dir, output_name, num_workers, batch_size, device, jit_path):
    emb = ImageEmbedding(device=device,jit_path=jit_path)
    bim = BatchInferModel(emb, device) 
    assert os.path.isdir(output_dir)
    
    txds = TxDataset(dataset, tx=preprocess)
    dl = DataLoader(txds, num_workers=num_workers, shuffle=False,
                    batch_size=batch_size, collate_fn=iden)
    res = []
    for batch in tqdm(dl):
        flat_batch = sum(batch,[])
        batch_res = bim(flat_batch)
        res.append(batch_res)

    merged_res = pd.concat(res, ignore_index=True)
    merged_res.to_parquet(f'{output_dir}/{output_name}.parquet')

# def preprocess_dataset(*, image_src, seesaw_root, dataset_name):
#     mp.set_start_method('spawn')
#     gdm = GlobalDataManager(seesaw_root)
#     gdm.create_dataset(image_src=image_src, dataset_name=dataset_name)
#     mc = gdm.get_dataset(dataset_name)
#     mc.preprocess()

import pyarrow
from pyarrow import parquet as pq
import shutil

import io

class SeesawDatasetManager:
    def __init__(self, root, dataset_name):
        ''' Assumes layout created by create_dataset
        '''
        self.dataset_name = dataset_name
        self.dataset_root = f'{root}/{dataset_name}'
        file_meta = pd.read_parquet(f'{self.dataset_root}/file_meta.parquet')
        self.file_meta = file_meta
        self.paths = file_meta['file_path'].values
        self.image_root = f'{self.dataset_root}/images'

    def get_pytorch_dataset(self):
        return ExplicitPathDataset(root_dir=self.image_root, relative_path_list=self.paths)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.dataset_name})'
        
    def preprocess(self, model_path,num_subproc=-1, force=False):
        ''' Will run a preprocess pipeline and store the output
            -1: use all gpus
            0: use one gpu process and preprocessing all in the local process 
        '''
        dataset = self.get_pytorch_dataset()
        jit_path=model_path
        vector_root = self.vector_path()
        if os.path.exists(vector_root):
            i = 0
            while True:
                i+=1
                backup_name = f'{vector_root}.bak.{i:03d}'
                if os.path.exists(backup_name):
                    continue
                else:
                    os.rename(vector_root, backup_name)
                    break
        
        os.makedirs(vector_root, exist_ok=False)

        if num_subproc == -1:
            num_subproc = torch.cuda.device_count()
        
        jobs = []
        if num_subproc == 0:
            worker_function(0, dataset, slice_size=len(dataset), 
                indices=np.arange(len(dataset)), cpus_per_proc=0, vector_root=vector_root, jit_path=jit_path)
        else:
            indices = np.random.permutation(len(dataset))
            slice_size = int(math.ceil(indices.shape[0]/num_subproc))
            cpus_per_proc = min(os.cpu_count()//num_subproc,4) if num_subproc > 0 else 0

            for i in range(num_subproc):
                p = mp.Process(target=worker_function, args=(i,dataset, slice_size, indices, cpus_per_proc, vector_root, jit_path))
                jobs.append(p)
                p.start()

            for p in jobs:
                p.join()

    def preprocess2(self, model_path):
        dataset = self.get_pytorch_dataset()
        jit_path=model_path
        vector_root = self.vector_path()
        if os.path.exists(vector_root):
            i = 0
            while True:
                i+=1
                backup_name = f'{vector_root}.bak.{i:03d}'
                if os.path.exists(backup_name):
                    continue
                else:
                    os.rename(vector_root, backup_name)
                    break
        
        os.makedirs(vector_root, exist_ok=False)

        sds = self
        paths = ((sds.image_root + '/' + sds.paths)).tolist()
        paths = [os.path.realpath(p) for p in paths]
        meta_dict = dict(zip(paths,zip(sds.paths,np.arange(len(sds.paths)))))

        ngpus = round(ray.available_resources()['GPU'])
        actors = [ray.remote(Preprocessor).options(num_cpus=5, num_gpus=1).remote(jit_path=jit_path, 
        output_dir=vector_root, meta_dict=meta_dict) for i in range(ngpus)]
        #pool = ray.util.ActorPool(actors)

        rds = (ray.data
                    .read_binary_files(paths=paths, include_paths=True, parallelism=400)
                    .split(ngpus,locality_hints=actors)
        )

        # indices = np.random.permutation(len(dataset))
        # slice_size =  2000
        res_iter = []
        for (actor,shard) in zip(actors,rds):
            of = actor.extract_meta.remote(shard)
            res_iter.append(of)
        # res_iter = pool.map_unordered(
        #     lambda a, idxs: a.extract_meta.remote(dataset, idxs),
        #     [indices[start_range:start_range+slice_size] for start_range in range(0,len(indices),slice_size)]
        # )
        ray.get(res_iter)
        # ofiles = []
        # for o in tqdm(res_iter):
        #     assert o not in ofiles
        #     ofiles.append(o)
        

    def save_vectors(self, vector_data):
        assert (np.sort(vector_data.dbidx.unique()) == np.arange(self.paths.shape[0])).all()
        vector_root = self.vector_path()
        os.makedirs(vector_root, exist_ok=False)
        vector_data.to_parquet(f'{vector_root}/manually_saved_vectors.parquet')

    def save_ground_truth(self, box_data, qgt=None):
        """ 
            Will add qgt and box information. or overwrite it.
        """
        if qgt is None:
            qgt = infer_qgt_from_boxes(box_data, num_files=self.paths.shape[0])

        assert qgt.shape[0] == self.paths.shape[0]
        gt_root = self.ground_truth_path()
        os.makedirs(gt_root, exist_ok=True)
        box_data.to_parquet(f'{gt_root}/boxes.parquet')
        qgt.to_parquet(f'{gt_root}/qgt.parquet')

    def vector_path(self):
        return f'{self.dataset_root}/meta/vectors'

    def ground_truth_path(self):
        gt_root = f'{self.dataset_root}/ground_truth/'
        return gt_root

    def load_vec_table(self, columns='default'):
        if columns == 'default':
            columns = ['dbidx', 'zoom_level', 'x1', 'y1', 'x2', 'y2','vectors']
        elif columns is None:
            pass
        else:
            assert False

        tab = pq.read_table(self.vector_path(),use_pandas_metadata=False, 
            memory_map=True, )
        df = tab.to_pandas(deduplicate_objects=False, ignore_metadata=True)

        x = df
        # x = x.assign(vectors=TensorArray(x['vectors'].to_numpy().astype('half')))
        return x
    
    def load_evdataset(self) -> EvDataset:
        df = self.load_vec_table(columns='default')
        fine_grained_meta = df[['dbidx', 'zoom_level', 'x1', 'y1', 'x2', 'y2']]
        fine_grained_embedding = df['vectors'].values.to_numpy()
        qgt = pd.read_parquet(f'{self.dataset_root}/ground_truth/qgt.parquet')
        box_data = pd.read_parquet(f'{self.dataset_root}/ground_truth/boxes.parquet')

        coarse_vecs = fine_grained_meta[fine_grained_meta.zoom_level == 1]
        embedded_dataset=infer_coarse_embedding(df[['dbidx', 'zoom_level', 'vectors']])

        return EvDataset(root=self.image_root, paths=self.paths, 
            embedded_dataset=embedded_dataset, 
            query_ground_truth=qgt, 
            box_data=box_data, 
            embedding=None,#model used for embedding 
            fine_grained_embedding=fine_grained_embedding,
            fine_grained_meta=fine_grained_meta)



def convert_dbidx(ev : EvDataset, ds : SeesawDatasetManager, prepend_ev :str = ''):
    new_path_df = ds.file_meta.assign(dbidx=np.arange(ds.file_meta.shape[0]))
    old_path_df = pd.DataFrame({'file_path':prepend_ev + ev.paths , 'dbidx':np.arange(len(ev.paths))})
    ttab = pd.merge(new_path_df, old_path_df, left_on='file_path', right_on='file_path', suffixes=['_new', '_old'], how='outer')
    assert ttab[ttab.dbidx_new.isna()].shape[0] == 0
    tmp = pd.merge(ev.box_data, ttab[['dbidx_new', 'dbidx_old']], left_on='dbidx', right_on='dbidx_old', how='left')
    tmp = tmp.assign(dbidx=tmp.dbidx_new)
    new_box_data = tmp[[c for c in tmp if c not in ['dbidx_new', 'dbidx_old']]]
    ds.save_ground_truth(new_box_data)

def infer_qgt_from_boxes(box_data, num_files):
    qgt = box_data.groupby(['dbidx', 'category']).size().unstack(level=1).fillna(0)
    qgt = qgt.reindex(np.arange(num_files)).fillna(0)
    return qgt.clip(0,1)

def infer_coarse_embedding(pdtab):
    max_zoom_out = pdtab.groupby('dbidx').zoom_level.max().rename('max_zoom_level')
    wmax = pd.merge(pdtab, max_zoom_out, left_on='dbidx', right_index=True)
    lev1 = wmax[wmax.zoom_level == wmax.max_zoom_level]
    res = lev1.groupby('dbidx').vectors.mean().values.to_numpy()
    normres = res/np.maximum(np.linalg.norm(res, axis=1,keepdims=True), 1e-6)
    return normres


"""
Expected data layout for a seesaw root with a few datasets:
/workdir/seesaw_data
├── coco  ### not yet preproceseed, just added
│   ├── file_meta.parquet
│   ├── images -> /workdir/datasets/coco
├── coco5 ### preprocessed
│   ├── file_meta.parquet
│   ├── images -> /workdir/datasets/coco
│   └── meta
│       └── vectors
│           ├── part_000.parquet
│           └── part_001.parquet
├── mini_coco
│   ├── file_meta.parquet
│   ├── images -> /workdir/datasets/mini_coco/images
│   └── meta
│       └── vectors
│           ├── part_000.parquet
│           └── part_001.parquet
"""
class GlobalDataManager:
    def __init__(self, root):
        if not os.path.exists(root):
            print('creating new data folder')
            os.makedirs(root)
        else:
            assert os.path.isdir(root)
            
        self.root = root
        
    def list_datasets(self):
        return [f for f in os.listdir(self.root) if not f.startswith('_')]
    
    def create_dataset(self, image_src, dataset_name, paths=[]) -> SeesawDatasetManager:
        '''
            if not given explicit paths, it assumes every jpg, jpeg and png is wanted
        '''
        assert ' ' not in dataset_name
        assert '/' not in dataset_name
        assert not dataset_name.startswith('_')
        assert dataset_name not in self.list_datasets(), 'dataset with same name already exists'
        image_src = os.path.realpath(image_src)
        assert os.path.isdir(image_src)
        dspath = f'{self.root}/{dataset_name}'
        assert not os.path.exists(dspath), 'name already used'
        os.mkdir(dspath)
        image_path = f'{dspath}/images'
        os.symlink(image_src, image_path)
        if len(paths) == 0:
            paths = list_image_paths(image_src)
            
        df = pd.DataFrame({'file_path':paths})
        df.to_parquet(f'{dspath}/file_meta.parquet')
        return self.get_dataset(dataset_name)
            
    def get_dataset(self, dataset_name) -> SeesawDatasetManager:
        assert dataset_name in self.list_datasets(), 'must create it first'
        ## TODO: cache this representation
        return SeesawDatasetManager(self.root, dataset_name)

    def clone(self, ds = None, ds_name = None, clone_name : str = None) -> SeesawDatasetManager:
        assert ds is not None or ds_name is not None
        if ds is None:
            ds = self.get_dataset(ds_name)

        if clone_name is None:
            dss = self.list_datasets()
            for i in range(len(dss)):
                new_name = f'{ds.dataset_name}_clone_{i:03d}'
                if new_name not in dss:
                    clone_name = new_name
                    break

        assert clone_name is not None
        shutil.copytree(src=ds.dataset_root, dst=f'{self.root}/{clone_name}', symlinks=True)
        return self.get_dataset(clone_name)

    def clone_subset(self, ds=None, ds_name=None, 
                subset_name : str = None, file_names=None, file_indices=None) -> SeesawDatasetManager:

        assert ds is not None or ds_name is not None
        if ds is None:
            ds = self.get_dataset(ds_name)

        dataset = ds
        # 1: names -> indices
        image_src = os.path.realpath(dataset.image_root)
        assert subset_name not in self.list_datasets(), 'dataset already exists'

        if file_names is not None:
            paths= dataset.file_meta.file_path.values
            dbidx = range(len(paths))
            lookup = dict(zip(paths,dbidx))
            file_indices = np.array([lookup[fn] for fn in file_names])
        elif file_indices is not None:
            file_indices = np.array(file_indices)
            file_names = dataset.file_meta.file_path.values[file_indices]
        else:
            assert False, 'need one of file_names or file_indices'
        
        self.create_dataset(image_src=image_src, dataset_name=subset_name, paths=file_names)
        subds = self.get_dataset(subset_name)

        id_set = pr.BitMap(file_indices)
        old_ids = file_indices
        new_ids = np.arange(len(file_indices))
        id_map = dict(zip(old_ids,new_ids))

        ## TODO: handle cases where no vecs/gt exist
        if os.path.exists(dataset.vector_path()):
            vt = dataset.load_vec_table()
            vt = vt[vt.dbidx.isin(id_set)]
            ## remap ids:
            vt = vt.assign(dbidx=vt.dbidx.map(lambda old_id : id_map[old_id]))
            subds.save_vectors(vt)

        if os.path.exists(dataset.ground_truth_path()):
            boxes = pd.read_parquet(f'{dataset.ground_truth_path()}/boxes.parquet')
            boxes = boxes[boxes.dbidx.isin(id_set)]
            boxes = boxes.assign(dbidx=boxes.dbidx.map(lambda old_id : id_map[old_id]))
            
            qgt = pd.read_parquet(f'{dataset.ground_truth_path()}/qgt.parquet')
            qgt = qgt.iloc[file_indices]
            qgt = qgt.reset_index(drop=True)

            subds.save_ground_truth(box_data=boxes, qgt=qgt)

        return self.get_dataset(subset_name)
    

    def __repr__(self):
        return f'{self.__class__.__name__}({self.root})'


def prep_ground_truth(paths, box_data, qgt):
    """adds dbidx column to box data, sets dbidx in qgt and sorts qgt by dbidx
    """
    orig_box_data = box_data
    orig_qgt = qgt
    
    path2idx = dict(zip(paths, range(len(paths))))
    mapfun = lambda x : path2idx.get(x,-1)
    box_data = box_data.assign(dbidx=box_data.file_path.map(mapfun))
    box_data = box_data[box_data.dbidx >= 0]
    
    new_ids = qgt.index.map(mapfun)
    qgt = qgt[new_ids >= 0]
    qgt = qgt.set_index(new_ids[new_ids >= 0])
    qgt = qgt.sort_index()
    
    assert len(paths) == qgt.shape[0], 'every path should be in the ground truth'
    return box_data, qgt