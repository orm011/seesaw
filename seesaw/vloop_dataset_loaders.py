from .embeddings import *
from .cross_modal_db import EmbeddingDB
import json
import pandas as pd
import os
from .ui.widgets import MImageGallery
from .fine_grained_embedding import *


class ExplicitPathDataset(object):
    def __init__(self, root_dir, relative_path_list):
        '''
        Reads images in a directory according to an explicit list.
        '''
        self.root = root_dir
        self.paths = relative_path_list
        self.formatter = 'http://localhost:9000/{root}/{path}'.format

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        relpath = self.paths[idx]
        return PIL.Image.open('{}/{}'.format(self.root, relpath))

    def get_urls(self, idxs):
        idxs = np.array(idxs).astype('int')
        return [self.formatter(root=self.root, path=rp) for rp in self.paths[idxs]]

    def show_images(self, idxbatch, values=None):
        ds = self
        idxbatch = np.array(idxbatch)
        ldata = []
        urls = ds.get_urls(idxbatch)
        
        dat = {'image_urls':urls, 'ldata':ldata}
        
        if values is None:
            values = np.ones_like(idxbatch)*-1
        
        for idx,val in zip(idxbatch, values):
            ldata.append({'dbidx':int(idx), 'boxes':[], 'value':int(val)})

        pn = MImageGallery(**dat)
        return pn


import inspect, os, copy

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)[:,np.newaxis]
    return vecs/(norms + 1e-6)

def load_vecs(path):
    vecs = np.load(path)
    check_vecs(vecs)
    return vecs

def check_vecs(vecs):
    assert vecs.dtype == 'float32'
    samp = np.random.permutation(len(vecs))[:1000] # don't run it on everything.
    vecs = vecs[samp]
    norms = np.linalg.norm(vecs, axis=-1)
    assert np.isclose(norms, 1., atol=1e-5, rtol=1e-5).all(), (norms.min(), norms.max())

def get_qgt(box_data, min_box_size):
    box_data = box_data.assign(width=box_data.x2 - box_data.x1, height = box_data.y2 - box_data.y1)
    box_data = box_data[(box_data.width >= min_box_size) & (box_data.height >= min_box_size)]
    frame_gt = box_data.groupby(['dbidx', 'category']).size().unstack(-1)
    frame_gt = frame_gt.fillna(0).clip(0,1)
    return frame_gt

def make_evdataset(*, root, paths, embedded_dataset, query_ground_truth, box_data, embedding, 
                fine_grained_embedding=None, fine_grained_meta=None, min_box_size=0.):

    #if query_ground_truth is None:
    #query_ground_truth = get_qgt(box_data, min_box_size) # boxes are still there 
    # for use by algorithms but not used for accuracy metrics
    query_ground_truth = query_ground_truth.clip(0.,1.)
    # assert query_ground_truth.max() == 1.
    # assert query_ground_truth.min() == 0.

    #root = os.path.abspath(root)
    check_vecs(embedded_dataset)

    if fine_grained_embedding is not None:
        check_vecs(fine_grained_embedding)

    return EvDataset(root=root, paths=paths, embedded_dataset=embedded_dataset, 
            query_ground_truth=query_ground_truth, 
            box_data=box_data, embedding=embedding, 
            fine_grained_embedding=fine_grained_embedding, fine_grained_meta=fine_grained_meta)

## make a very cheap ev dataset that can be sliced cheaply

class EvDataset(object):
    query_ground_truth : pd.DataFrame
    box_data : pd.DataFrame
    embedding : XEmbedding
    embedded_dataset : np.ndarray
    fine_grained_embedding : np.ndarray
    fine_grained_meta : pd.DataFrame
    imge_dataset : ExplicitPathDataset

    def __init__(self, *, root, paths, embedded_dataset, query_ground_truth, box_data, embedding, 
                fine_grained_embedding=None, fine_grained_meta=None):
        """
        meant to be cheap to run, we use it to make subsets.
            do any global changes outside, eg in make_evdataset.
        """
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        kwargs = {k:v for (k,v) in values.items() if k in args}
        self.__dict__.update(kwargs)
        self.image_dataset = ExplicitPathDataset(root_dir=root, relative_path_list=paths)
        
    def __len__(self):
        return len(self.image_dataset)

def get_class_ev(ev, category, boxes=False): 
    """restrict dataset to only those indices labelled for a given category"""
    gt = ev.query_ground_truth[category]
    class_idxs = gt[~gt.isna()].index.values
    ev1 = extract_subset(ev, idxsample=class_idxs, categories=[category],boxes=boxes)
    return ev1, class_idxs

def extract_subset(ev : EvDataset, *, idxsample=None, categories=None, boxes=True) -> EvDataset:
    """makes an evdataset consisting of those categories and those index samples only
    """
    if categories is None:
        categories = ev.query_ground_truth.columns.values

    if idxsample is None:
        idxsample = np.arange(len(ev), dtype='int')
    else:
        assert (np.sort(idxsample) == idxsample).all(), 'sort it because some saved data assumed it was sorted here'

    categories = set(categories)
    idxset = pr.BitMap(idxsample)

    qgt = ev.query_ground_truth[categories]
    embedded_dataset = ev.embedded_dataset
    paths = ev.paths

    if not (idxset == pr.BitMap(range(len(ev)))): # special case where everything is copied
        qgt = qgt.iloc[idxsample].reset_index(drop=True)
        embedded_dataset = embedded_dataset[idxsample]
        paths = ev.paths[idxsample]

    if boxes:
        good_boxes = ev.box_data.dbidx.isin(idxset) & ev.box_data.category.isin(categories)
        if good_boxes.all(): # avoid copying when we want everything anyway (common case)
            sample_box_data = ev.box_data
        else:
            sample_box_data = ev.box_data[good_boxes]
            lookup_table = np.zeros(len(ev), dtype=np.int) - 1
            lookup_table[idxsample] = np.arange(idxsample.shape[0], dtype=np.int)
            sample_box_data = sample_box_data.assign(dbidx = lookup_table[sample_box_data.dbidx])
            assert (sample_box_data.dbidx >= 0 ).all()
    else:
        sample_box_data = None

    if ev.fine_grained_embedding is not None:
        meta, vec = restrict_fine_grained(ev.fine_grained_meta,ev.fine_grained_embedding, idxsample)
    else:
        meta = None
        vec = None

    return EvDataset(root=ev.root, paths=paths,
                    embedded_dataset=embedded_dataset,
                    query_ground_truth=qgt,
                    box_data=sample_box_data,
                    embedding=ev.embedding,
                    fine_grained_embedding=vec,
                    fine_grained_meta=meta)

def bdd_full(embedding : XEmbedding):
    image_root = './data/bdd_root/images/100k/'
    paths = np.load('./data/bdd_valid_paths.npy', allow_pickle=True)
    box_data = pd.read_parquet('./data/bdd_boxes_all_qgt_classes_imsize.parquet')
    qgt = pd.read_parquet('./data/bdd_boxes_all_qgt_classes_qgt.parquet')


    #nm.to_parquet('./data/bdd_multiscale_meta_all2.parquet')
    #np.save('./data/bdd_multiscale_vecs_all2_normalized_float16.npy',nv.astype('float16'))
    fgmeta = pd.read_parquet('./data/bdd_multiscale_meta_all2.parquet')
    fge = load_vecs('./data/bdd_multiscale_vecs_all2_normalized_float32.npy')
    assert fgmeta.shape[0] == fge.shape[0]
    # fge = np.load('./data/bdd_multiscale_vecs_all2_normalized_float32.npy', mmap_mode='r')

    # fgmeta = pd.read_parquet('./data/bdd_20k_finegrained_acc_meta.parquet')
    # fge = np.load('./data/bdd_20k_finegrained_acc.npy')

    # embedded_dataset = np.load('./data/bdd_all_valid_feats_CLIP.npy', mmap_mode='r')
    embedded_dataset = load_vecs('./data/bdd_all_valid_feats_CLIP_normalized_float32.npy')
    # embedded_dataset = np.load('./data/bdd_all_valid_feats_CLIP_normalized_float32.npy', mmap_mode='r')
    ev1 = make_evdataset(root=image_root, paths=paths, 
                         embedded_dataset=embedded_dataset,
                         query_ground_truth=qgt,
                         box_data=box_data, embedding=embedding, 
                         fine_grained_embedding=fge,
                         fine_grained_meta=fgmeta)

    return ev1

def objectnet_cropped(embedding : XEmbedding) -> EvDataset:
    image_vectors = load_vecs('./data/objnet_cropped_CLIP_normalized_float32.npy')
    #np.load('./data/objnet_cropped_CLIP.npy', mmap_mode='r')
    tmp = np.load('./data/objnet_vectors_cropped.npz', allow_pickle=True)
    paths = tmp['paths']
    root = './data/objectnet/images/'## lost cropped copy with drive
    dir2cat = json.load(open('./data/folder_to_objectnet_label.json'))
    categories = list(map(lambda x : dir2cat[x.split('/')[0]].lower(), paths))

    df = pd.DataFrame({'idx':np.arange(len(categories)), 'path':paths,'category':categories})
    query_ground_truth = df.groupby(['idx', 'path', 'category']).size().unstack(level=-1).fillna(0)
    query_ground_truth = query_ground_truth.reset_index(drop=True)
    box_data = df.assign(dbidx=df.idx, x1=0, x2=224, y1=0, y2=224, im_width=224, im_height=224)

    # emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    ## objectnet is a special case with only one vector per image.
    fge = image_vectors
    fgmeta = pd.DataFrame({'dbidx':np.arange(len(fge))})
    fgmeta['iis'] = 0
    fgmeta['jjs'] = 0
    fgmeta['zoom_level'] = 0
    fgmeta['zoom_factor'] = 1.
    
    return make_evdataset(root=root, 
                     paths=paths, 
                     embedded_dataset=image_vectors, 
                     query_ground_truth=query_ground_truth, 
                     box_data=box_data, 
                     embedding=embedding, 
                     fine_grained_embedding=fge,
                     fine_grained_meta=fgmeta)

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

# def lvis_data():
#     root = './data/lvis/'
#     coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')
#     coco_files = coco_files.reset_index().rename(mapper={'index':'dbidx'}, axis=1)
#     paths = coco_files['paths'].values
#     return ExplicitPathDataset(root, paths)

def lvis_full(embedding : XEmbedding) -> EvDataset:
    # NB. unlike other datasets with only a handful of categories,
    # for each category, livs annotates a different subset of the data, and leaves
    # a large subset unlabelled.
    # positive and one negative subset,
    # for synonyms and definitions etc. can use: './data/lvis_trainval_categories.parquet'
    qgt = pd.read_parquet('./data/lvis_val_query_ground_truth.parquet')
    bd = pd.read_parquet('./data/lvis_boxes_imsize.parquet')
    paths = np.load('./data/lvis_paths.npy', allow_pickle=True) # should be same as coco paths.

    # bd = pd.read_parquet('./data/lvis_boxes_wdbidx.parquet')    
    # emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    # pd.read_parquet('./data/lvis_val_categories.parquet')
    #emb_vectors = np.load('./data/coco_full_CLIP.npy', mmap_mode='r')
    emb_vectors = load_vecs('./data/coco_full_CLIP_normalized_float32.npy')
    #emb_vectors = np.load('./data/coco_full_CLIP_normalized_float32.npy',mmap_mode='r')
    root = './data/coco_root/' # using ./data/lvis/ seems to be slower for browser url loading.
    # coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')
    # coco_files = coco_files.reset_index().rename(mapper={'index':'dbidx'}, axis=1)
    # paths = coco_files['paths'].values

    gvec_meta = pd.read_parquet('./data/lvis_multiscale_meta_all.parquet')
    gvecs = load_vecs('./data/lvis_multiscale_vecs_all_normalized_float32.npy')#, mmap_mode='r')
    # gvec_meta = pd.read_parquet('./data/lvis_finegrained_acc_meta.parquet')
    # gvecs = np.load('./data/lvis_finegrained_acc.npy')

    return make_evdataset(root=root, 
                     paths=paths, 
                     embedded_dataset=emb_vectors, 
                     query_ground_truth=qgt, 
                     box_data=bd, 
                     embedding=embedding, fine_grained_embedding=gvecs,
                     fine_grained_meta=gvec_meta)

def lvis_category(embedding: XEmbedding, category : str) -> EvDataset:
    ''' makes dataset for a single lvis category (using only the labelled subset)
    '''
    ev0 = lvis_full(embedding)
    ev, class_idxs = get_class_ev(ev0, category, boxes=True)
    return ev

def coco_full(embedding : XEmbedding) -> EvDataset:

    emb_vectors = load_vecs('./data/coco_full_CLIP_normalized_float32.npy')
    # emb_vectors = np.load('./data/coco_full_CLIP.npy', mmap_mode='r')
    root = './data/coco_root/'
    paths = np.load('./data/coco_paths.npy', allow_pickle=True)
    box_data = pd.read_parquet('./data/coco_boxes_imsize.parquet')
    qgt = pd.read_parquet('./data/coco_full_qgt.parquet')

    # coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')
    # coco_files = coco_files.reset_index().rename(mapper={'index':'dbidx'}, axis=1)
    # paths = coco_files['paths'].values
    
    # ## box data
    # box_data = pd.read_parquet('./data/coco_boxes_all_no_crowd.parquet')
    # coco_files = pd.read_parquet('./data/coco_full_CLIP_paths.parquet')

    # coco_files = coco_files.reset_index(drop=False)
    # coco_files = coco_files.rename(mapper={'index':'dbidx'}, axis=1)

    # box_data = box_data.merge(coco_files, left_on='image_id', right_on='coco_id', how='inner')
    # xmin = box_data.bbox.apply(lambda x : x[0])
    # ymin = box_data.bbox.apply(lambda x : x[1])
    # w = box_data.bbox.apply(lambda x : x[2])
    # h = box_data.bbox.apply(lambda x : x[3])
    # box_data = box_data.assign(x1 = xmin, y1 = ymin, x2 = xmin + w, y2=ymin + h, w=w, h=h)

    # ## need query ground truth
    # coco_val = json.load(open('./data/coco_root/annotations/instances_val2017.json'))
    # id2name = {c['id']:c['name'] for c in coco_val['categories']}

    # qgt = box_data.groupby(['dbidx', 'category_id']).size().gt(0).astype('float').unstack(level='category_id')
    # qgt = qgt.reindex(np.arange(paths.shape[0])) # fill images with no boxes
    # qgt = qgt.fillna(0)
    # qgt = qgt.rename(mapper=id2name, axis=1)
    # qgt = qgt.clip(0,1)
    # nm.to_parquet('./data/lvis_multiscale_meta_all.parquet')
    # np.save('./data/lvis_multiscale_vecs_all_normalized_float16.npy',nv.astype('float16'))
    #
    gvec_meta = pd.read_parquet('./data/lvis_multiscale_meta_all.parquet')
    gvecs = load_vecs('./data/lvis_multiscale_vecs_all_normalized_float32.npy')
    # gvec_meta = pd.read_parquet('./data/lvis_finegrained_acc_meta.parquet')
    # gvecs = np.load('./data/lvis_finegrained_acc.npy',mmap_mode='r')

    return make_evdataset(root=root, paths=paths, embedded_dataset=emb_vectors, 
                     query_ground_truth=qgt, box_data=box_data, embedding=embedding, fine_grained_embedding=gvecs,
                     fine_grained_meta=gvec_meta)
    

def coco_split(coco_full : EvDataset, split : str) -> EvDataset:
    paths = coco_full.paths
    assert split in ['val', 'train']
    val_set_dbidx = np.where(np.array(list(map(lambda x : x.startswith('val'), paths))))[0]
    train_set_dbidx = np.where(np.array(list(map(lambda x : x.startswith('train'), paths))))[0]

    idxs = { 'val':val_set_dbidx, 'train':train_set_dbidx }
    return extract_subset(coco_full, idxsample=idxs[split])


def mini_coco(embedding : XEmbedding) -> EvDataset:
    image_root = './data/mini_coco/images/'
    relpaths = np.load('./data/mini_coco/relpaths.npy', allow_pickle=True)
    box_data = pd.read_parquet('./data/mini_coco/box_data.parquet')
    embedded_dataset = np.load('./data/mini_coco/embedded_dataset.npy')
    fine_grained_embedding = np.load('./data/mini_coco/fine_grained_embedding.npy')
    fine_grained_meta = pd.read_parquet('./data/mini_coco/fine_grained_meta.parquet')

    qgt = box_data.groupby(['dbidx', 'category']).size().unstack(level=1).fillna(0)
    # assert qgt.shape[0] == relpaths.shape[0]


    return make_evdataset(root=image_root, paths=relpaths, embedded_dataset=embedded_dataset, 
                     query_ground_truth=qgt, box_data=box_data, embedding=embedding,fine_grained_embedding=fine_grained_embedding,
                     fine_grained_meta=fine_grained_meta)




def dota1_full(embedding : XEmbedding) -> EvDataset:
    root = './data/dota_dataset/'    
    relpaths= np.load('./data/dota1_paths.npy', allow_pickle=True)
    embedded_vecs = load_vecs('./data/dota_224_pool_clip_normalized_float32.npy')#, mmap_mode='r')
    box_data = pd.read_parquet('./data/dota1_boxes_imsize.parquet')
    # box_data = pd.read_parquet('./data/dota1_boxes.parquet')
    # pngs = pd.Categorical(box_data.relpath, ordered=True).dtype.categories.map(lambda x : x.replace('labelTxt-v1.0', 'images/images').replace('.txt', '.png'))
    # relpaths = pngs.values
    # box_data = box_data.assign(box_id=np.arange(box_data.shape[0]))
    # box_data = box_data.assign(x1=box_data.xmin, x2=box_data.xmax, y1=box_data.ymin, y2=box_data.ymax)
    # box_data  = box_data.assign(category=box_data.cat)
    # box_data = box_data[['x1', 'y1', 'x2', 'y2', 'relpath', 'dbidx', 'category', 'relpath', 'is_difficult']]
    qgt = box_data.groupby(['dbidx', 'category']).size().unstack(level=1).fillna(0)
    assert qgt.shape[0] == relpaths.shape[0]
    # qgt = qgt.clip(0,1)#gt(0).astype('float')        
    # embedded_dataset = embedding.from_image(img_vec=embedded_vecs)
    fgmeta = pd.read_parquet('./data/dota_multiscale_meta_all.parquet')
    fge = load_vecs('./data/dota_multiscale_vecs_all_normalized_float32.npy')
    # fge = np.load('./data/dota_finegrained_acc.npy',mmap_mode='r')
    # fgmeta = pd.read_parquet('./data/dota_finegrained_acc_meta.parquet')

    return make_evdataset(root=root, paths=relpaths, embedded_dataset=embedded_vecs, 
                     query_ground_truth=qgt, box_data=box_data, embedding=embedding,fine_grained_embedding=fge,
                     fine_grained_meta=fgmeta)

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
    qgt = an1.reindex(np.arange(ava_paths.shape[0])).fillna(0)
    # qgt = qgt.clip(0,1)#.gt(0).astype('float')
    categories = qgt.columns
    
    emb_vectors = embedding.from_image(img_vec=image_vectors.astype('float32'))
    
    return make_evdataset(root=root, 
                     paths=ava_paths, 
                     embedded_dataset=emb_vectors, 
                     query_ground_truth=qgt, 
                     box_data=ava_boxes, 
                     embedding=embedding)