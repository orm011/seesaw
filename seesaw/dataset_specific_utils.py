from tqdm.auto import tqdm
import glob
import json
import pandas as pd

def parse_bdd_annotations(base : str, label_set='det_20'):
    """ Reads and flattens BDD label files into two pandas dataframes,
        one holding image per row information, 
        and one holding one-box per row information. 
        (some images have no boxes)
        Attribute names are preserved.
        Can handle either the detection labels or the object tracking labels.
    """
    assert label_set in ['det_20', 'box_track_20']
    
    rows = []
    frames = []
    
    files = glob.glob(f'{base}/{label_set}/**/*json', recursive=True) # 2 for the det-20 files, and can be many for the mot-20 files

    for f in tqdm(files):
        full = json.load(open(f, 'r'))
        annfile = f[len(base)+1:] # the part of the files not from the base
        for obj in full:
            if 'labels' in obj: # not every frame has object labels
                labels = obj['labels']
                del obj['labels']
            else:
                labels = []

            if 'attributes' in obj: # frames in tracking dataset do not have this
                frame_attributes = obj['attributes']
                del obj['attributes']
            else:
                frame_attributes = {}

            frames.append({'annotation_file':annfile, 'num_labels':len(labels), **obj, **frame_attributes})

            for lab in labels:
                if 'attributes' in lab:
                    label_attributes = lab['attributes']
                    del lab['attributes']

                else:
                    label_attributes = {}

                box = lab['box2d']
                del lab['box2d']

                row = {'annotation_file':annfile, **obj, **lab, **label_attributes, **box}
                rows.append(row)
    
    boxdf = pd.DataFrame(rows)
    framedf = pd.DataFrame(frames)
    return boxdf, framedf
