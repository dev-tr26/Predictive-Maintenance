# if equipment will fail within 30 cycles window

from __future__ import annotations
import numpy as np
import joblib
from xgboost import XGBClasssifier 

DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    reg_alpha=0.0,
    scale_pos_weight=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

class FailureClassifier:
    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}
        self.model = XGBClasssifier(**self.params)
    
    def fit(self, X_train,y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train,y_train, eval_set=eval_set,verbose=False)
        return self
    
    def predict_prob(self, X)->np.ndarray:
        return self.model.predict_prob(X)[:,1]
    
    def predict(self,X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_prob(X) >= threshold).astype(int)
    
    def feature_importances(self, feature_names):
        imp = self.model.feature_importances_
        return sorted(zip(feature_names, imp), key=lambda t: -t[1])
    
    def save(self, path:str):
        joblib.dump({"model": self.model, "params":self.params}, path)
        
        
    @classmethod
    def load(cls, path: str) -> "FailureClassifier":
        payload = joblib.load(path)
        obj = cls(**payload["params"])
        obj.model = payload["model"]
        return obj 
    
