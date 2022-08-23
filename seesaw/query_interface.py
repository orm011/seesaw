import pyroaring as pr
from .indices.interface import get_constructor, AccessMethod
from .basic_types import *
import pandas as pd


class LabelDB:
    def __init__(self):
        self.ldata = {}

    def get_seen(self):
        return pr.BitMap(self.ldata.keys())

    def put(self, dbidx: int, boxes: List[Box]):
        self.ldata[dbidx] = boxes

    def get(self, dbidx: int, format: str):
        dbidx = int(dbidx)
        if dbidx not in self.ldata:
            return None  # has not been seen. used when sending data

        boxes = self.ldata[dbidx]
        if boxes is None:  # seen by user but not labeled. consider negative for now
            boxes = []

        if format == "df":
            if boxes == []:
                return pd.DataFrame(boxes, columns=["x1", "x2", "y1", "y2"]).astype(
                    "float32"
                )  # cols in case it is empty
            else:
                return pd.DataFrame([b.dict() for b in boxes])[
                    ["x1", "x2", "y1", "y2"]
                ].astype("float32")
        elif format == "box":
            return boxes
        elif format == 'binary':
            if len(boxes) == 0:
                return 0
            else:
                return 1
        else:
            assert False, 'unknown format'

class InteractiveQuery(object):
    """
    tracks what has been shown already and supplies it
    to stateless db as part of query
    """

    def __init__(self, index: AccessMethod):
        self.index = index
        self.returned = (
            pr.BitMap()
        )  # images returned from index (not necessarily seen yet)
        self.label_db = LabelDB()
        self.startk = 0

    def query_stateful(self, *args, **kwargs):
        """
        :param kwargs: forwards arguments to db query method but
         also keeping track of already seen results. also
         keeps track of query history.
        :return:
        """
        batch_size = kwargs.get("batch_size")
        del kwargs["batch_size"]

        res = self.index.query(
            *args, topk=batch_size, **kwargs, exclude=self.returned, startk=self.startk
        )
        if 'excluded_dbidxs' in res.keys(): 
            self.returned.update(res['excluded_dbidxs'])
            del res['excluded_dbidxs']
        else: 
            self.returned.update(res["dbidxs"])
        # assert nextstartk >= self.startk nor really true: if vector changes a lot,
        self.startk = res["nextstartk"]
        del res["nextstartk"]
        return res

    def getXy(self):
        raise NotImplementedError("abstract")
