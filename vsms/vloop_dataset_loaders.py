from .embeddings import *
from .cross_modal_db import EmbeddingDB
import json
import pandas as pd
import os

class ExplicitPathDataset(object):
    def __init__(self, root_dir, relative_path_list):
        '''
        Reads images in a directory according to an explicit list.
        '''
        self.root = os.path.abspath(root_dir)
        self.paths = relative_path_list
        self.formatter = 'http://clauslocal:8000/{root}/{path}'.format

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        relpath = self.paths[idx]
        return PIL.Image.open('{}/{}'.format(self.root, relpath))

    def get_urls(self, idxs):
        idxs = np.array(idxs).astype('int')
        return [self.formatter(root=self.root, path=rp) for rp in self.paths[idxs]]


class TxDataset(object):
    def __init__(self, ds, tx):
        self.ds = ds
        self.tx = tx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.tx(self.ds[idx])

import inspect, os, copy
class EvDataset(object):
    def __init__(self, *, root, paths, embedded_dataset, query_ground_truth, box_data, embedding):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        kwargs = {k:v for (k,v) in values.items() if k in args}
        self.__dict__.update(kwargs)
        self.image_dataset = ExplicitPathDataset(root_dir=root, relative_path_list=paths)
        abs_root = os.path.abspath(root)
        self.db = EmbeddingDB(self.image_dataset, embedding, embedded_dataset)
        
    def __len__(self):
        return len(self.image_dataset)

    def copy(self):
        return EvDataset(root=copy.copy(self.root), 
            paths=self.paths.copy(),
            embedded_dataset=self.embedded_dataset.copy(),
            query_ground_truth=self.query_ground_truth.copy(),
            box_data=self.box_data.copy() if self.box_data is not None else None,
            embedding=self.embedding)

def extract_subset(ev : EvDataset, idxsample, categories='all', boxes=True) -> EvDataset:
    """makes an evdataset consisting of that index sample only
    """
    if categories == 'all':
        categories = list(ev.query_ground_truth.columns.values)

    if boxes:
        perm = np.argsort(idxsample)
        idxsample = idxsample[perm] # box code assumes sorted indices
        # idxsample = np.sort(idxsample) # box code assumes sorted indices
        new_pos = np.zeros(ev.paths.shape[0]).astype('int')
        new_pos[idxsample] = 1
        new_pos[idxsample] = new_pos.cumsum().astype('int')[idxsample]
        new_pos = new_pos - 1
        random2new = new_pos

        good_boxes = ev.box_data.dbidx.isin(idxsample) & ev.box_data.category.isin(categories)
        sample_box_data = ev.box_data[good_boxes]
        sample_box_data = sample_box_data.assign(dbidx = random2new[sample_box_data.dbidx])
        assert (sample_box_data.dbidx >= 0 ).all()
    else:
        sample_box_data = None


    return EvDataset(root=ev.root, paths=ev.paths[idxsample],
                    embedded_dataset=ev.embedded_dataset[idxsample],
                    query_ground_truth=ev.query_ground_truth[categories].iloc[idxsample].reset_index(drop=True),
                    box_data=sample_box_data,
                    embedding=ev.embedding)

def bdd_full(embedding : XEmbedding, embedded_vecs : np.ndarray = None) -> EvDataset:
    bdd_data = dict(embedding_path='./data/bdd_all_feats.npy',
                    embedding_file_list='./data/bdd_all_file_validity.parquet',
                    data_root='./data/bdd_root/',
                    ground_truth_path='./data/bdd_query_binary_ground_truth.parquet',
                    box_path='./data/bdd100k_all_boxes_and_polygons.parquet',
                    # './data/bdd100k_labels_images_scene_attr.parquet' for attributes
                    name='bdd'
                    )

    bddpaths = pd.read_parquet('./data/bdd_all_file_validity.parquet')
    bddpaths = bddpaths.assign(file=bddpaths.relpath.map(lambda x: '/'.join(x.split('/')[-2:])))
    bddpaths = bddpaths.reset_index()
    bddpaths = bddpaths.assign(vector_index=bddpaths['index'])

    image_root = './data/bdd_root/images/100k/'

    bddgt = pd.read_parquet('./data/bdd_query_binary_ground_truth.parquet')
    filecol = bddgt.reset_index().reset_index()
    filecol = filecol.assign(gt_index=filecol['index'])
    filecol = filecol[['file', 'gt_index']]

    corr = pd.merge(filecol, bddpaths[['file', 'vector_index']], left_on='file', right_on='file', how='inner')
    paths = corr.file.values
    filtered_gt = bddgt.iloc[corr.gt_index].reset_index(drop=True)

    if embedded_vecs is None:
        if isinstance(embedding, CLIPWrapper):
            filtered_vecs = np.load('./data/bdd_all_valid_feats_CLIP.npy')
        else:
            bdd_feats = np.load('./data/bdd_all_feats.npy')
            filtered_vecs = bdd_feats[corr.vector_index]
    else:
        filtered_vecs = embedded_vecs

    boxgt = pd.read_parquet(bdd_data['box_path'])
    bdd_boxes = boxgt[boxgt.annotation_type == 'box']
    bdd_boxes = bdd_boxes.assign(file=bdd_boxes.file.map(lambda x: x.split('/', 1)[-1]))
    box_data = bdd_boxes.merge(pd.DataFrame({'file': paths,
                                             'dbidx': np.arange(paths.shape[0])}),
                               left_on='file', right_on='file', how='right')

    embedded_dataset = embedding.from_image(img_vec=filtered_vecs)
    return EvDataset(root=image_root, paths=corr.file.values, 
                     embedded_dataset=embedded_dataset,
                     query_ground_truth=filtered_gt,
                     box_data=box_data, embedding=embedding)

def objectnet_cropped(embedding : XEmbedding, embedded_vecs : np.array = None ) -> EvDataset:
    image_vectors = np.load('./data/objnet_cropped_CLIP.npy')
    tmp = np.load('./data/objnet_vectors_cropped.npz', allow_pickle=True)
    paths = tmp['paths']
    root = './data/objectnet/cropped/'
    dir2cat = json.load(open('./data/objectnet/mappings/folder_to_objectnet_label.json'))
    categories = list(map(lambda x : dir2cat[x.split('/')[0]].lower(), paths))

    df = pd.DataFrame({'idx':np.arange(len(categories)), 'path':paths,'category':categories})
    query_ground_truth = df.groupby(['idx', 'path', 'category']).size().unstack(level=-1).fillna(0)
    query_ground_truth = query_ground_truth.reset_index(drop=True)
    box_data = df.assign(dbidx=df.idx, x1=10, x2=224-10, y1=10, y2=224-10)

    emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    
    return EvDataset(root=root, 
                     paths=paths, 
                     embedded_dataset=emb_vectors, 
                     query_ground_truth=query_ground_truth, 
                     box_data=box_data, 
                     embedding=embedding)

def coco2lvis(coco_name):
    '''mapping of coco category names to the best approximate lvis names.
    '''
    coco2lvis = {
        # the rest are lvis_name = coco_name.replace(' ', '_')
        'car':'car_(automobile)',
        'bus':'bus_(vehicle)',
        'train':'train_(railroad_vehicle)',
        'fire hydrant':'fireplug',
        'tie':'necktie',
        'skis':'ski',
        'sports ball':'ball',
        'wine glass':'wineglass',
        'orange':'orange_(fruit)',
        'hot dog':'sausage',
        'donut':'doughnut',
        'couch':'sofa',
        'potted plant':'flowerpot',
        'tv':'television_set',
        'laptop':'laptop_computer',
        'mouse':'mouse_(computer_equipment)',
        'remote':'remote_control', 
        'keyboard':'computer_keyboard', ## check if this is what coco means.
        'cell phone':'cellular_telephone', 
        'microwave':'microwave_oven',
        'hair drier':'hair_dryer'
    }


    lvis_name = coco2lvis.get(coco_name, coco_name.replace(' ', '_'))
    lvis_name = lvis_name.replace('_', ' ') # only needed bc lvis qgt did this.
    return lvis_name

def lvis_data():
    root = './data/lvis/'
    coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')
    coco_files = coco_files.reset_index().rename(mapper={'index':'dbidx'}, axis=1)
    paths = coco_files['paths'].values
    return ExplicitPathDataset(root, paths)


def lvis_full(embedding : XEmbedding, embedded_vecs : np.array = None) -> EvDataset:
    # if embedded_vecs is None and isinstance(embedding, CLIPWrapper):
    #     image_vectors = np.load('./data/coco_full_CLIP.npy')
    # else:
    image_vectors = embedded_vecs

    # NB. unlike other datasets with only a handful of categories,
    # for each category, livs annotates a different subset of the data, and leaves
    # a large subset unlabelled.
    # positive and one negative subset,
    # for synonyms and definitions etc. can use: './data/lvis_trainval_categories.parquet'

    qgt = pd.read_parquet('./data/lvis_val_query_ground_truth.parquet')
    #qgt = pd.read_parquet('./data/lvis_coco_merged_query_ground_truth.parquet')
    bd = pd.read_parquet('./data/lvis_boxes_wdbidx.parquet')    
    emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    #pd.read_parquet('./data/lvis_val_categories.parquet')
    qgt = qgt.clip(0.,1.)
    root = './data/coco_root/' # using ./data/lvis/ seems to be slower for browser url loading.
    coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')
    coco_files = coco_files.reset_index().rename(mapper={'index':'dbidx'}, axis=1)
    paths = coco_files['paths'].values

    return EvDataset(root=root, 
                     paths=paths, 
                     embedded_dataset=emb_vectors, 
                     query_ground_truth=qgt, 
                     box_data=bd, 
                     embedding=embedding)


def coco_full(embedding : XEmbedding, embedded_vecs : np.ndarray = None) -> EvDataset:

    # if embedded_vecs is None:
    #     if isinstance(embedding, CLIPWrapper):
    image_vectors = np.load('./data/coco_full_CLIP.npy')
    #     else:
    #         assert False, 'other model not computed for full coco'
    # else:
    # image_vectors = embedded_vecs 

    
    root = './data/coco_root/'
    coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')
    coco_files = coco_files.reset_index().rename(mapper={'index':'dbidx'}, axis=1)
    paths = coco_files['paths'].values
    emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    
    ## box data
    box_data = pd.read_parquet('./data/coco_boxes_all_no_crowd.parquet')
    coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')

    coco_files = coco_files.reset_index(drop=False)
    coco_files = coco_files.rename(mapper={'index':'dbidx'}, axis=1)

    box_data = box_data.merge(coco_files, left_on='image_id', right_on='coco_id', how='inner')
    xmin = box_data.bbox.apply(lambda x : x[0])
    ymin = box_data.bbox.apply(lambda x : x[1])
    w = box_data.bbox.apply(lambda x : x[2])
    h = box_data.bbox.apply(lambda x : x[3])
    box_data = box_data.assign(x1 = xmin, y1 = ymin, x2 = xmin + w, y2=ymin + h, w=w, h=h)


    ## need query ground truth
    coco_val = json.load(open('./data/coco_root/annotations/instances_val2017.json'))
    id2name = {c['id']:c['name'] for c in coco_val['categories']}

    qgt = box_data.groupby(['dbidx', 'category_id']).size().gt(0).astype('float').unstack(level='category_id')
    qgt = qgt.reindex(np.arange(paths.shape[0])) # fill images with no boxes
    qgt = qgt.fillna(0)
    qgt = qgt.rename(mapper=id2name, axis=1)

    embedded_dataset = embedding.from_image(img_vec=image_vectors)
    return EvDataset(root=root, paths=paths, embedded_dataset=embedded_dataset, 
                     query_ground_truth=qgt, box_data=box_data, embedding=embedding)
    

def coco_split(coco_full : EvDataset, split : str) -> EvDataset:
    paths = coco_full.paths
    assert split in ['val', 'train']
    val_set_dbidx = np.where(np.array(list(map(lambda x : x.startswith('val'), paths))))[0]
    train_set_dbidx = np.where(np.array(list(map(lambda x : x.startswith('train'), paths))))[0]

    idxs = { 'val':val_set_dbidx, 'train':train_set_dbidx }
    return extract_subset(coco_full, idxs[split])


def dota1_full(embedding : XEmbedding, embedded_vecs : np.ndarray = None) -> EvDataset:
    root = './data/dota_dataset/'    
    # if embedded_vecs is None:
    #     assert isinstance(embedding, CLIPWrapper)
    embedded_vecs = np.load('./data/dota_224_pool_clip.npy')
        
    box_data = pd.read_parquet('./data/dota1_boxes.parquet')
    pngs = pd.Categorical(box_data.relpath, ordered=True).dtype.categories.map(lambda x : x.replace('labelTxt-v1.0', 'images/images').replace('.txt', '.png'))
    relpaths = pngs.values
    box_data = box_data.assign(box_id=np.arange(box_data.shape[0]))
    box_data = box_data.assign(x1=box_data.xmin, x2=box_data.xmax, y1=box_data.ymin, y2=box_data.ymax)
    box_data  = box_data.assign(category=box_data.cat)
    box_data = box_data[['x1', 'y1', 'x2', 'y2', 'relpath', 'dbidx', 'category', 'relpath', 'is_difficult']]

    qgt = box_data.groupby(['dbidx', 'category']).size().unstack(level=1).fillna(0).gt(0).astype('float')        
    embedded_dataset = embedding.from_image(img_vec=embedded_vecs)
    return EvDataset(root=root, paths=relpaths, embedded_dataset=embedded_dataset, 
                     query_ground_truth=qgt, box_data=box_data, embedding=embedding)

def ava22(embedding : XEmbedding, embedded_vecs : np.ndarray = None) -> EvDataset:
    # if isinstance(embedding, CLIPWrapper):
    image_vectors = np.load('./data/ava_dataset_embedding.npy')
    # else:
    #     assert False, 'other model not computed for full lvis'
    root = './data/ava_dataset/images/'
    ava_files = pd.read_parquet('./data/ava_dataset_image_paths.parquet')
    ava_boxes = pd.read_parquet('./data/ava_dataset_annotations2.parquet')
    
    ava_boxes = ava_boxes.rename(mapper={'action_name':'category'},axis=1)
    ava_paths = ava_files['image_file'].values
    an1 = ava_boxes.groupby(['dbidx', 'category']).size().unstack(level=-1)
    qgt = an1.reindex(np.arange(ava_paths.shape[0])).fillna(0).gt(0).astype('float')
    categories = qgt.columns
    
    emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    
    return EvDataset(root=root, 
                     paths=ava_paths, 
                     embedded_dataset=emb_vectors, 
                     query_ground_truth=qgt, 
                     box_data=ava_boxes, 
                     embedding=embedding)