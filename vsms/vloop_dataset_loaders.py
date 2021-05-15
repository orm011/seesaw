from .embeddings import *
from .cross_modal_db import EmbeddingDB
import json
import pandas as pd

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
        relpath = self.paths[idx]
        return PIL.Image.open('{}/{}'.format(self.root, relpath))

class TxDataset(object):
    def __init__(self, ds, tx):
        self.ds = ds
        self.tx = tx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.tx(self.ds[idx])

import inspect, os
class EvDataset(object):
    def __init__(self, *, root, paths, embedded_dataset, query_ground_truth, box_data, embedding):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        kwargs = {k:v for (k,v) in values.items() if k in args}
        self.__dict__.update(kwargs)
        self.image_dataset = ExplicitPathDataset(root_dir=root, relative_path_list=paths)
        abs_root = os.path.abspath(root)
        url_template = 'http://clauslocal:8000/{abs_root}/{rel_path}'
        urls = [url_template.format(abs_root=abs_root, rel_path=rp) for rp in paths]
        self.db = EmbeddingDB(self.image_dataset, embedding, embedded_dataset, urls=urls)
        
    def __len__(self):
        return len(self.image_dataset)

def extract_subset(ev : EvDataset, idxsample) -> EvDataset:
    """makes an evdataset consisting of that index sample only
    """
    idxsample = np.sort(idxsample) # box code assumes sorted indices
    new_pos = np.zeros(ev.paths.shape[0]).astype('int')
    new_pos[idxsample] = 1
    new_pos[idxsample] = new_pos.cumsum().astype('int')[idxsample]
    new_pos = new_pos - 1
    random2new = new_pos

    good_boxes = ev.box_data.dbidx.isin(idxsample)
    sample_box_data = ev.box_data[good_boxes]
    sample_box_data = sample_box_data.assign(dbidx = random2new[sample_box_data.dbidx])
    assert (sample_box_data.dbidx >= 0 ).all()

    return EvDataset(root=ev.root, paths=ev.paths[idxsample],
                    embedded_dataset=ev.embedded_dataset[idxsample],
                    query_ground_truth=ev.query_ground_truth.iloc[idxsample].reset_index(drop=True),
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

def objectnet_cropped(embedding : XEmbedding, embedded_vecs : np.array ) -> EvDataset:
    tmp = np.load('./data/objnet_vectors_cropped.npz', allow_pickle=True)
    paths = tmp['paths']
    root = './data/objectnet/cropped/'
    dir2cat = json.load(open('./data/objectnet/mappings/folder_to_objectnet_label.json'))
    categories = list(map(lambda x : dir2cat[x.split('/')[0]].lower(), paths))

    df = pd.DataFrame({'idx':np.arange(len(categories)), 'path':paths,'category':categories})
    query_ground_truth = df.groupby(['idx', 'path', 'category']).size().unstack(level=-1).fillna(0)
    query_ground_truth = query_ground_truth.reset_index(drop=True)
    box_data = df.assign(dbidx=df.idx, x1=10, x2=224-10, y1=10, y2=224-10)
    
    if isinstance(embedding,CLIPWrapper):
        image_vectors = np.load('./data/objnet_cropped_CLIP.npy')
    else:
        image_vectors = tmp['vecs']

    emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    
    return EvDataset(root=root, 
                     paths=paths, 
                     embedded_dataset=emb_vectors, 
                     query_ground_truth=query_ground_truth, 
                     box_data=box_data, 
                     embedding=embedding)

def lvis_full(embedding : XEmbedding, embedded_vecs : np.array) -> EvDataset:
    if embedded_vecs is None and isinstance(embedding, CLIPWrapper):
        image_vectors = np.load('./data/coco_full_CLIP.npy')
    else:
        image_vectors = embedded_vecs

    root = './data/lvis/'
    coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')
    coco_files = coco_files.reset_index().rename(mapper={'index':'dbidx'}, axis=1)
    paths = coco_files['paths'].values

    box_data = pd.read_parquet('./data/lvis_annotations_all.parquet')
    xmin = box_data.bbox.apply(lambda x : x[0])
    ymin = box_data.bbox.apply(lambda x : x[1])
    w = box_data.bbox.apply(lambda x : x[2])
    h = box_data.bbox.apply(lambda x : x[3])
    
    query_ground_truth = pd.read_parquet('./data/lvis_val_query_ground_truth.parquet')
    categories = query_ground_truth.columns
    bd = box_data.assign(x1 = xmin, y1 = ymin, x2 = xmin + w, y2=ymin + h, w=w, h=h)
    bd = bd.merge(coco_files[['dbidx', 'coco_id']], left_on='image_id', right_on='coco_id', how='left')
    bd = bd.assign(category=bd.category_id.map(lambda catid : categories[catid - 1]))
    
    emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    
    return EvDataset(root=root, 
                     paths=paths, 
                     embedded_dataset=emb_vectors, 
                     query_ground_truth=query_ground_truth, 
                     box_data=bd, 
                     embedding=embedding)


def coco_full(embedding : XEmbedding, embedded_vecs : np.ndarray = None) -> EvDataset:

    if embedded_vecs is None:
        if isinstance(embedding, CLIPWrapper):
            image_vectors = np.load('./data/coco_full_CLIP.npy')
        else:
            assert False, 'other model not computed for full coco'
    else:
         image_vectors = embedded_vecs 

    
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
    if embedded_vecs is None:
        assert isinstance(embedding, CLIPWrapper)
        embedded_vecs = np.load('./data/dota_224_pool_clip.npy')
        
    box_data = pd.read_parquet('./data/dota1_boxes.parquet')
    qgt = box_data.groupby(['dbidx', 'cat']).size().unstack(level=1).fillna(0).gt(0).astype('float')    
    pngs = pd.Categorical(box_data.relpath, ordered=True).dtype.categories.map(lambda x : x.replace('labelTxt-v1.0', 'images/images').replace('.txt', '.png'))
    relpaths = pngs.values
    box_data = box_data.assign(box_id=np.arange(box_data.shape[0]))
    box_data = box_data.assign(x1=box_data.xmin, x2=box_data.xmax, y1=box_data.ymin, y2=box_data.ymax)
    box_data  = box_data.assign(category=box_data.cat)
    box_data = box_data[['x1', 'y1', 'x2', 'y2', 'relpath', 'dbidx', 'category', 'relpath', 'is_difficult']]
    box_data = box_data[box_data.is_difficult == 0]
    
    embedded_dataset = embedding.from_image(img_vec=embedded_vecs)
    return EvDataset(root=root, paths=relpaths, embedded_dataset=embedded_dataset, 
                     query_ground_truth=qgt, box_data=box_data, embedding=embedding)

def ava22(embedding : XEmbedding, embedded_vecs : np.ndarray = None ) -> EvDataset:
    if isinstance(embedding, CLIPWrapper):
        image_vectors = np.load('./data/ava_dataset_embedding.npy')
    else:
        assert False, 'other model not computed for full lvis'

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