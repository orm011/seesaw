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