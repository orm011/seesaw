import numpy as np
import pyroaring as pr

# instead of EvDataset
# ev.embedding.embed_str: abstract away string emb.
# ev.image_dataset #potentially used for some methods that will look at the image based on feedback
# AccessMethod() # 
# Query() # keeps annotations, keeps list of previously seen, etc. 

class AccessMethod:
    def string2vec(self, string : str) -> np.ndarray:
        raise NotImplementedError('implement me')

    def query(self, *, vector : np.ndarray, topk : int, exclude : pr.BitMap = None) -> np.ndarray:
        raise NotImplementedError('implement me')

    def new_query(self):
        raise NotImplementedError('implement me')

class InteractiveQuery(object):
    """
        tracks what has been shown already and supplies it 
        to stateless db as part of query
    """
    def __init__(self, db: AccessMethod):
        self.db = db
        self.seen = pr.BitMap()
        self.acc_idxs = []
        self.startk = 0

    def query_stateful(self, *args, **kwargs):
        '''
        :param kwargs: forwards arguments to db query method but
         also keeping track of already seen results. also
         keeps track of query history.
        :return:
        '''
        batch_size = kwargs.get('batch_size')
        del kwargs['batch_size']
            
        idxs, nextstartk = self.db.query(*args, topk=batch_size, **kwargs, exclude=self.seen, startk=self.startk)
        # assert nextstartk >= self.startk nor really true: if vector changes a lot, 
        self.startk = nextstartk
        self.seen.update(idxs)
        self.acc_idxs.append(idxs)
        return idxs, nextstartk