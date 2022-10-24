from .basic_types import Box, List
import pandas as pd
import pyroaring as pr

class LabelDB:
    def __init__(self):
        self.ldata = {}

    def get_seen(self):
        return pr.BitMap(self.ldata.keys())

    def put(self, dbidx: int, boxes: List[Box]):
        self.ldata[dbidx] = boxes

    def get_box_df(self):
        empty_df = pd.DataFrame([], columns=["dbidx", "x1", "x2", "y1", "y2"]).astype("float32")
        empty_df = empty_df.assign(dbidx=empty_df.dbidx.astype('int32'))

        dfs = [ empty_df ]

        for dbidx,v in self.ldata.items():
            if v == []:
                continue
            
            df = pd.DataFrame([b.dict() for b in v])[
                    ["x1", "x2", "y1", "y2"]].astype("float32")
            df = df.assign(dbidx=dbidx)
            df = df.assign(dbidx=df.dbidx.astype('int32'))
            dfs.append(df)

        c = pd.concat(dfs, ignore_index=True)
        return c


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