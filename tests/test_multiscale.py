from seesaw.indices.multiscale.multiscale_index import box_iou
import pandas as pd
import numpy as np

def test_box_iou():
    dfempty = pd.DataFrame({'x1':[], 'x2':[], 'y1':[], 'y2':[]})
    df1 = pd.DataFrame({'x1':[10], 'x2':[20], 'y1':[10], 'y2':[20]})
    df2 = pd.DataFrame({'x1':[10, 20], 'x2':[30, 30], 'y1':[10, 20], 'y2':[30, 30]})

    iou1, containment1 = box_iou(dfempty, df2, return_containment=True)
    assert iou1.shape == (0,2)
    assert containment1.shape == iou1.shape

    iou2, containment2 = box_iou(df1, df2, return_containment=True)
    assert iou2.shape == (1,2)
    assert containment2.shape == iou2.shape
    assert np.isclose(iou2[0,0], .25)
    assert np.isclose(iou2[0,1], 0)

    assert np.isclose(containment2[0,0], 1.) # fully contained
    assert np.isclose(containment2[0,1], 0.) # fully disjoint