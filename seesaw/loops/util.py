import numpy as np
import os

def makeXy(idx, lr, sample_size, pseudoLabel=True):
    is_labeled = lr.is_labeled > 0
    X = idx.vectors[is_labeled]
    y = lr.labels[is_labeled]
    is_real = np.ones_like(y)
    
    if pseudoLabel:
        vec2 = idx.vectors[~is_labeled]
        ylab2 = lr.current_scores()[~is_labeled]
        rsample = np.random.permutation(vec2.shape[0])[:sample_size]

        Xsamp = vec2[rsample]
        ysamp = ylab2[rsample]
        is_real_samp = np.zeros_like(ysamp)
        
        X = np.concatenate((X, Xsamp))
        y = np.concatenate((y, ysamp))
        is_real = np.concatenate((is_real, is_real_samp))
        
    return X,y,is_real


def get_image_paths(image_root, path_array, idxs):
    return [
        os.path.normpath(f"{image_root}/{path_array[int(i)]}").replace("//", "/")
        for i in idxs
    ]

def clean_path(path):
    return os.path.normpath(os.path.abspath(os.path.realpath(path)))



