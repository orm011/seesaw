from .box_utils import BoundingBoxBatch, Segment
import pandas as pd
import numpy as np

def transpose_test_case(case):
    box, cont, soln = case
    tbox = [box[1], box[0], box[3], box[2]]
    tcont = [cont[1], cont[0]]
    tsoln = [soln[1], soln[0], soln[3], soln[2]]
    return (tbox, tcont, tsoln)

def get_test_cases():
    base_cases = [ # x1 y1 x2 y2 im_w im_h
        ([10, 10,  20, 30], [100, 50], [ 5,10,25,30]),# fits comfortably
        ([10, 10,  30, 20], [100, 50], [10, 5,30,25]),# fits comfortably
        ([ 0, 10, 100, 40], [100, 50], [25, 0,75,50]),# too wide, many solutions, pick center
        ([40,  0,  60, 50], [100, 50], [25, 0,75,50]),# very tall, but should fit
        ([ 0,  0,  10, 20], [100, 50], [ 0, 0,20,20]),# near top left boundary. fit around, not centered
        ([ 100-10,  50-20,  100, 50], [100, 50], [ 100-20, 50-20, 100,50]),# near bottom right boundary. fit around, not centered
    ]

    transposed_cases = list(map(transpose_test_case, base_cases))
    test_cases = base_cases + transposed_cases

    test_df = pd.DataFrame([box + cont  for (box, cont, _) in test_cases], columns=['x1', 'y1', 'x2', 'y2', 'im_width', 'im_height'])
    soln_df = pd.DataFrame([soln + cont for (_, cont, soln) in test_cases], columns=['x1', 'y1', 'x2', 'y2','im_width', 'im_height'])
    return test_df, soln_df

def test_convert():
    # check conversion works without losing info
    test_df, _ = get_test_cases()
    bbx = BoundingBoxBatch.from_dataframe(test_df)
    assert np.isclose(bbx.to_xyxy(), test_df[['x1', 'y1', 'x2', 'y2']].values).all()

def test_square_box():
    test_df, soln_df = get_test_cases()
    bbdf = BoundingBoxBatch.from_dataframe(test_df)
    sqbx = bbdf.best_square_box()
    assert np.isclose(sqbx.to_xyxy(),  soln_df[['x1', 'y1', 'x2', 'y2']].values).all()

def test_square_box_min_size(): 
    ''' test case where desired min size is too large for image (eg small images) '''
    test_df, soln_df = get_test_cases()
    bbdf = BoundingBoxBatch.from_dataframe(test_df)
    sqbx = bbdf.best_square_box(min_side=2) # small side should have no effect on boxes already larger
    assert np.isclose(sqbx.to_xyxy(),  soln_df[['x1', 'y1', 'x2', 'y2']].values).all()

    sqbx_large = bbdf.best_square_box(min_side=1000) # very large min_side should work and make boxes be the minimum side
    assert np.isclose(sqbx_large.width(), np.minimum(soln_df.im_width, soln_df.im_height)).all()
    assert np.isclose(sqbx_large.height(), np.minimum(soln_df.im_width, soln_df.im_height)).all()

def test_segment_intersect():
    base_cases = [ # a , b, intersection, disjoint
        [(0, 10), (5, 15), (5, 10), False], # partial
        [(0, 20), (5, 15), (5, 15), False], # contained
        [(0, 5), (10, 15), (0, 0), True], # disjoint, limit values do not matter
    ]

    flipped_cases = [ (b,a, c,d) for (a,b,c,d) in base_cases]

    test_cases = base_cases + flipped_cases

    segsA = Segment.from_x1x2(x1x2=np.array([ a for (a,_,_,_) in test_cases]))
    segsB = Segment.from_x1x2(x1x2=np.array([ b for (_,b,_,_) in test_cases]))
    expected = Segment.from_x1x2(x1x2=np.array([ c for (_,_,c, _) in test_cases]))
    is_disjoint = np.array([c for (_,_,_,c) in test_cases])
    intersect = segsA.intersection(segsB)

    zeros = np.isclose(intersect.length(), 0) 
    assert (zeros == is_disjoint).all()

    assert (intersect.to_x1x2()[~is_disjoint] == expected.to_x1x2()[~is_disjoint]).all()
