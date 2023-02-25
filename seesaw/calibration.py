import numpy as np
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration

class BasicModel:
    ### dummy model sticking to sklearn interface
    def __init__(self, vec):
        self.coeff_ = vec.reshape(-1)
        self.classes_ = [0,1]
        
    def fit(self, X,y):
        pass

    def predict_proba(self, X):
        p =  X @ self.coeff_
        return np.stack([1-p, p], axis=-1)

def compute_calibrated_probabilities(vector_scorer, X,  y):
    assert X.shape[1] == vector_scorer.reshape(-1).shape[0]
    assert X.shape[0] == y.shape[0]
    
    model = BasicModel(vector_scorer)
    ccv = CalibratedClassifierCV(base_estimator=model, cv='prefit')
    ccv.fit(X, y)
    return ccv.predict_proba(X)[:,1]


class Calibrator:
    def __init__(self, X, y):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self._mean = y.mean()

    def get_mean(self):
        return self._mean

    def get_probabilities(self, vector_scorer, vectors):
        ### fit with given labels then apply to given vectors.
        sc = _SigmoidCalibration()
        raw_scores = self.X @ vector_scorer.reshape(-1)
        sc.fit(raw_scores.reshape(-1,1), self.y)

        infer_scores = vectors @ vector_scorer.reshape(-1)
        return sc.predict(infer_scores)