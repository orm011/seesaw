import numpy as np
import pandas as pd
import math

class Segment:
    """represents a batch of line 1-D segments"""
    def __init__(self, middle, radius): # use other methods
        self.middle : np.ndarray = middle
        self.radius : np.ndarray = radius
        
    @staticmethod
    def from_x1x2(*, x1x2 : np.ndarray  = None, x1 = None, x2 = None) -> 'Segment':
        if x1x2 is not None:
            x1 = x1x2[:,0]
            x2 = x1x2[:,1]
        else:
            assert x1 is not None and x2 is not None

        assert (x1 <= x2).all()
        
        mid = (x2 + x1)/2
        rad = (x2 - x1)/2
        return Segment(mid, rad)
    
    def to_x1x2(self) -> np.ndarray:
        return np.stack([self.x1(), self.x2()], axis=1)

    @staticmethod
    def from_midrad(mid, rad) -> 'Segment':
        assert (rad >= 0).all()
        return Segment(mid, rad)
        
    def mid(self) -> np.ndarray:
        return self.middle
        
    def rad(self) -> np.ndarray:
        return self.radius
    
    def x1(self) -> np.ndarray:
        return self.middle - self.radius
        
    def x2(self) -> np.ndarray:
        return self.middle + self.radius
        
    def clip(self, minx, maxx) -> 'Segment':
        minx = np.array(minx)
        maxx = np.array(maxx)
        assert (maxx >= minx).all()

        newx1 = np.clip(self.x1(), minx, None)
        newx2 = np.clip(self.x2(), None, maxx)
        return Segment.from_x1x2(x1=newx1, x2=newx2)

    def intersection(self, other : 'Segment') -> 'Segment':
        ## elementwise
        assert self.num_segments() == other.num_segments()
        x1tmp = np.maximum(self.x1(), other.x1())
        x2tmp = np.minimum(self.x2(), other.x2())

        x1 = np.clip(x1tmp, x1tmp, x2tmp) # make x1 <= x2
        return Segment.from_x1x2(x1=x1, x2=x2tmp)

    def num_segments(self) -> int:
        return self.middle.shape[0]

    def fits(self, minx=None, maxx=None):
        if minx is not None:
            c1 = self.x1() >= minx
        else:
            c1 = True
        
        if maxx is not None:
            c2 = self.x2() <= maxx
        else:
            c2 = True
        
        return (c1 & c2).all()
        
    def length(self) -> np.ndarray: 
        """ length of each segment """
        return 2*self.rad()

    def pad(self, padding, minx, maxx):
        padding = np.array(padding)
        assert (padding >= 0).all()

        return Segment.from_midrad(self.mid(), self.rad() + padding).clip(minx, maxx)

    def clip(self, minx, maxx) -> 'Segment':
        minx = np.array(minx)
        maxx = np.array(maxx)
        return Segment.from_x1x2(x1=np.clip(self.x1(), minx, maxx), 
                                x2=np.clip(self.x2(), minx, maxx))
    
    def best_seg(self, new_len, minx, maxx) -> 'Segment':
        """ forms a new segment `newseg` with following properties
        
            ## hard constraints
            newseg.length() == min(new_len, maxx - minx)
            minx <= newseg.x1() <= nnewseg.x2() <= maxx

            # newseg.intersect(self) is maximal
            
            ideally newseg.mid() == self.mid(), 
                but this isn't always possible (eg segment already near edge)
                so we aim to minimize |newseg.mid() - self.mid()|
            
            note new_len can be smaller or bigger than before.

        """
        minx = np.array(minx)
        maxx = np.array(maxx)
        assert (maxx >= minx).all()
        new_len = np.array(new_len)

        assert self.fits(minx, maxx)
        new_len = np.minimum(new_len, maxx - minx)
        
        raw_seg = Segment.from_midrad(self.mid(), new_len/2.)
        left_excess = np.clip(minx - raw_seg.x1(), 0, None)
        right_excess = np.clip(raw_seg.x2() - maxx, 0, None)

        assert (~((left_excess > 0) & (right_excess > 0))).all() # no excess could be both sides
        
        best_seg = Segment.from_midrad(mid=raw_seg.mid() + left_excess - right_excess, rad=raw_seg.rad())
        return best_seg.clip(minx, maxx) # sometimes there are small excesses, get rid of them.

def _as_polygons(df):
    import shapely
    import shapely.geometry

    df = df.assign(y1=-df.y1, y2=-df.y2) # reflect y-direction bc. svg 0,0 is bottom right
    return df[['x1', 'y1', 'x2', 'y2']].apply(lambda x : shapely.geometry.box(*tuple(x)), axis=1)


class BoxBatch:
    """represents a batch of rectangular boxes within a w x h image"""
    def __init__(self, xseg, yseg):
        self.xseg : Segment = xseg
        self.yseg : Segment = yseg
        
    @staticmethod
    def from_xyxy(xyxy : np.ndarray) -> 'BoxBatch':
        xseg = Segment.from_x1x2(x1x2=xyxy[:,[0,2]])
        yseg = Segment.from_x1x2(x1x2=xyxy[:,[1,3]])
        return BoxBatch(xseg, yseg)

    @staticmethod
    def from_dataframe(df : pd.DataFrame, xyxy_columns=['x1', 'y1', 'x2', 'y2']) -> 'BoxBatch':
        std_xyxy = ['x1', 'y1', 'x2', 'y2']
        tdict= dict(zip(xyxy_columns, std_xyxy))
        boxcols = df[xyxy_columns].rename(tdict, axis=1)
        return BoxBatch.from_xyxy(boxcols.values)

    def to_xyxy(self) -> np.ndarray:
        return np.stack([self.x1(), self.y1(), self.x2(), self.y2()], axis=1)

    def to_dataframe(self) -> pd.DataFrame:
        xyxy = self.to_xyxy()
        return pd.DataFrame({'x1':xyxy[:,0], 'y1':xyxy[:,1], 'x2':xyxy[:,2], 'y2':xyxy[:,3]})

    def __repr__(self) -> str:
        return self.to_dataframe().__repr__()

    def _repr_html_(self) -> str:
        df =  self.to_dataframe()
        polygons = _as_polygons(df) 
        df = df.assign(shape=polygons)
        styled = df.style.format({'shape':lambda shp: shp._repr_svg_()} , escape="html")
        return styled._repr_html_()

    def x1(self):
        return self.xseg.x1()
    
    def x2(self):
        return self.xseg.x2()
    
    def y1(self):
        return self.yseg.x1()
    
    def y2(self):
        return self.yseg.x2()
    
    def height(self):
        return self.yseg.length()
    
    def width(self):
        return self.xseg.length()

    def area(self):
        return self.height() * self.width()

    def intersection(self, other : 'BoxBatch') -> 'BoxBatch':
        return BoxBatch(xseg=self.xseg.intersection(other.xseg),
                        yseg=self.yseg.intersection(other.yseg))

    
    def pad(self, padding, xmax, ymax) -> 'BoxBatch':
        return BoxBatch(xseg=self.xseg.pad(padding, 0, xmax), 
                        yseg=self.yseg.pad(padding, 0, ymax))

    def best_square_box(self, xmax=math.inf, ymax=math.inf, min_side=0):
        """ gets the square box that fits within bounds, overlaps as much as possible with box,
            and is as near the center as possible"""

        xmax = np.array(xmax)
        ymax = np.array(ymax)
        min_side = np.array(min_side)
        max_container = np.minimum(xmax, ymax)
        box_side = np.maximum(self.height(), self.width())

        target_size = np.maximum(np.minimum(box_side, max_container), 
                                 np.minimum(min_side, max_container))

        new_yseg = self.yseg.best_seg(target_size, minx=0, maxx=ymax)
        new_xseg = self.xseg.best_seg(target_size, minx=0, maxx=xmax)
        return BoxBatch(new_xseg, new_yseg)


def _polygon_col(df, as_array=False):
    box_polygons = _as_polygons(df) # box polygons
    container_polygons = _as_polygons(pd.DataFrame({'x1':0, 'y1':0, 'x2':df['im_width'], 'y2':df['im_height']}))
    
    outarr = np.empty(len(box_polygons), dtype='object')
    for i, (bx,cont) in enumerate(zip(box_polygons, container_polygons)):
        outarr[i] = shapely.geometry.GeometryCollection([bx, cont.boundary])

    if as_array:
        return outarr

    geoms = pd.Series(data=outarr, index=df.index, dtype='object')
    return geoms

class BoundingBoxBatch(BoxBatch):
    """represents box batch in the context of a larger image  of size  w, h
    """
    def __init__(self, xseg, yseg, im_width, im_height):
        super().__init__(xseg, yseg)
        self.im_width : np.ndarray = np.array(im_width)
        self.im_height : np.ndarray = np.array(im_height)

    @staticmethod
    def from_boxbatch(bx, im_width, im_height):
        return BoundingBoxBatch(bx.xseg, bx.yseg, im_width, im_height)

    @staticmethod
    def from_dataframe(df, xyxy_columns=['x1', 'y1', 'x2', 'y2', 'im_height', 'im_width']) -> 'BoundingBoxBatch':
        std_xyxy = ['x1', 'y1', 'x2', 'y2', 'im_height', 'im_width']
        tdict= dict(zip(xyxy_columns, std_xyxy))
        boxcols = df[xyxy_columns].rename(tdict, axis=1)
        bb = BoxBatch.from_dataframe(boxcols)
        return BoundingBoxBatch(bb.xseg, bb.yseg, im_width=boxcols['im_width'].values, 
                                    im_height=boxcols['im_height'].values)
        
    def to_dataframe(self) -> pd.DataFrame:
        df = super().to_dataframe()
        return df.assign(im_width=self.im_width, im_height=self.im_height)

    def pad(self, padding):
        bbx = super().pad(padding=padding, xmax=self.im_width, ymax=self.im_height)
        return BoundingBoxBatch.from_boxbatch(bbx, self.im_width, self.im_height)

    def _repr_html_(self) -> str:
        df =  self.to_dataframe()
        df = df.assign(shape=_polygon_col(df))
        styled = df.style.format({'shape':lambda shp: shp._repr_svg_()} , escape="html")
        return styled._repr_html_()

    def best_square_box(self, min_side=0) -> 'BoundingBoxBatch':
        bb = super().best_square_box(xmax=self.im_width, ymax=self.im_height, min_side=min_side)
        return BoundingBoxBatch(bb.xseg, bb.yseg, self.im_width, self.im_height)        

class BoxOverlay:
    ''' overlays box information on image.
    '''
    def __init__(self, x1, y1, x2, y2, im_width, im_height, im_url=None, max_display_size=None):
        import shapely
        import shapely.geometry

        box = shapely.geometry.box(x1, y1, x2, y2) 
        
        self.im_width = im_width
        self.im_height = im_height

        if max_display_size is None:
            self.scale_factor = 1.
        else:
            self.scale_factor = round(min(max(im_width, im_height), max_display_size)/max(im_width, im_height), 2)
        
        container = shapely.geometry.box(0, 0, im_width, im_height)
        self.shape = shapely.geometry.GeometryCollection([box, container.boundary])
        self.im_url = im_url
        
    @staticmethod
    def from_dfrow(dfrow, im_url=None, max_display_size=None):
        return BoxOverlay(dfrow.x1, dfrow.y1, dfrow.x2, dfrow.y2, dfrow.im_width, 
                            dfrow.im_height, im_url=im_url, max_display_size=max_display_size)
    
    def _repr_html_(self):
        bxsvg = self.shape.svg()

        height = round(self.scale_factor*self.im_height)
        width = round(self.scale_factor*self.im_width)


        image_str = f'<img width="{width}" height="{height}" src="{self.im_url}"/>' if self.im_url else ''
        
        style_str = 'position:absolute;top:0;left:0' if self.im_url else '' # will show strangely in nb
        
        svg_str = f'''<svg style="{style_str}"
                        width="{width}" height="{height}" 
                        viewBox="0 0 {width} {height}">
                            <g transform="matrix({self.scale_factor:.02f},0,0,{self.scale_factor:.02f},0,0)">
                                {bxsvg}
                            </g>
                    </svg>
                  '''
        
        overlay = f'''<div style="position:relative;">
                        {image_str}
                        {svg_str}
                      </div>
                    '''
        
        return overlay


### some older methods we may want to fold above
def df2tensor(df1):
    import torch
    b1 = torch.from_numpy(
            np.stack([df1.x1.values, df1.y1.values, df1.x2.values, df1.y2.values], axis=1)
    )
    return b1

def box_iou(df1, df2, return_containment=False):
    import torchvision.ops
    b1 = df2tensor(df1)
    b2 = df2tensor(df2)

    inter, union = torchvision.ops.boxes._box_inter_union(b1, b2)
    b1_area = torchvision.ops.boxes.box_area(b1).reshape(-1, 1) # one per box

    ious = (inter/union).numpy()
    containment1 = (inter / b1_area).numpy() # debug orig_area
    
    if not return_containment:
        return ious
    else:
        return ious, containment1

def _iou_df(df1, df2, iou_min = 0):
    """ assumes df1 and df2 have boxes in them.
    """
    ious, cont  = box_iou(df1, df2, return_containment=True)
    # iis,jjs = np.meshgrid(np.arange(df1.shape[0]), np.arange(df2.shape[0]), indexing='ij')
    iis,jjs = np.where(ious > iou_min) # only return iou > 0
    #np.meshgrid(np.arange(df1.shape[0]), np.arange(df2.shape[0]), indexing='ij')
    return pd.DataFrame({'iloc_left':iis.reshape(-1),
                         'iloc_right':jjs.reshape(-1),
                         'iou':ious[iis,jjs],
                         'cont':cont[iis,jjs]})

def box_join(df1, df2, iou_gt=0):
    """ assumes df1 and df2 have box columns x1 x2 y1 y2. 
    """
    mg2 = _iou_df(df1, df2, iou_gt)
    suffixes = ['_left', '_right']
    z1 = df1.iloc[mg2.iloc_left.values].reset_index(drop=True).rename(lambda n : n + suffixes[0], axis=1)
    z2 = df2.iloc[mg2.iloc_right.values].reset_index(drop=True).rename(lambda n : n + suffixes[1], axis=1)
    joined = mg2.assign(**z1, **z2)
    return joined

def join_labels_single_frame(boxdf, labeldf, min_gt_contained):
    """ assigns labels from labeldf boxes to boxes in boxdf 
        based on whether the label box is substantially contained within the box.
        assumes all boxes and labels are for the same image
    """
    boxdf = boxdf[['box_id', 'x1', 'y1', 'x2', 'y2']]
    labeldf = labeldf[['category', 'box_id', 'x1', 'y1', 'x2', 'y2']]
    
    ious = _iou_df(labeldf, boxdf, min_gt_contained)
    labeldf = labeldf.assign(iloc_left=np.arange(labeldf.shape[0]))
    boxdf = boxdf.assign(iloc_right=np.arange(boxdf.shape[0]))

    mg1 = labeldf.merge(ious, on='iloc_left')
    mg2 = mg1.merge(boxdf[['box_id', 'iloc_right']], on='df2_iloc', suffixes=['', '_square'])
    mg2 = mg2.drop(['df1_iloc', 'df2_iloc'], axis=1)
    
    return mg2

def join_labels(boxdf, labeldf, min_gt_contained, sample_size=1):
    from tqdm.auto import tqdm
    gts = []
    boxgrps = boxdf.groupby('dbidx')
    for dbidx, gtgrp in tqdm(labeldf.groupby('dbidx')):
        boxgrp = boxgrps.get_group(dbidx)
        repbox = boxgrp.sample(n=sample_size)
        ngt = join_labels_single_frame(repbox, gtgrp, 
                                min_gt_contained=min_gt_contained)
        gts.append(ngt.assign(dbidx_square=dbidx))
        
    all_labels= pd.concat(gts, ignore_index=True)
    return all_labels

def left_iou_join(vector_meta_df, boxes):
    if vector_meta_df.shape[0] > 0:
        gps = []
        for dbidx, gp in vector_meta_df.groupby('dbidx'):
            anyboxes = boxes[boxes.dbidx == dbidx]
            max_iou = np.zeros(gp.shape[0])
            if anyboxes.shape[0] != 0:
                leftvals = box_join(gp, anyboxes[['x1', 'y1', 'x2', 'y2']]).groupby('iloc_left').iou.max()        
                max_iou[leftvals.index.values] = leftvals.values

            gp = gp.assign(max_iou=max_iou)
            gps.append(gp)
            
        return pd.concat(gps)
    else:
        return vector_meta_df.assign(max_iou=np.array([]).astype('float32'))