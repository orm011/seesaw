import pyroaring as pr
from .indices.interface import AccessMethod
from .labeldb import LabelDB

class InteractiveQuery(object):
    """
    tracks what has been shown already and supplies it
    to stateless db as part of query
    """

    def __init__(self, index: AccessMethod):
        self.index = index

        # images returned from index (not necessarily seen yet)
        self.returned = pr.BitMap() 

        # image labels received back
        self.label_db = LabelDB()

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
            *args, topk=batch_size, **kwargs, exclude=self.returned
        )
        
        self.returned.update(res["dbidxs"])
        return res

    def getXy(self, **options):
        raise NotImplementedError("abstract")
