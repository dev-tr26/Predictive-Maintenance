# loads trained artifacts once at API startup ans scores incoming readings 
# training pipeline uses rolling windows computed over a whole unit;s history 
# for live single reading inference we can;t recompute a pandas rolling window from scratch on each request 
# so unitstatebuffer is maintained for small per equipment ring buffer for recent raw readings in memory erives the identical rolling/EWMA/diff features causally from it


from __future__ import annotations
import json
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.models.xgb_classifier import FailureClassifier
from src.models.autoencoder import TorchAutoencoderAnomalyDetector
from src.preprocessing import SENSOR_COLS, SETTINGS_COLS, LOW_VARIANCE_SENSORS, SETTINGS_COLS
from src.feature_engineering import ROLLING_WINDOWS

ACTIVE_SENSORS = [c for c in SENSOR_COLS if c not in LOW_VARIANCE_SENSORS]
BUFFER_LEN = max(ROLLING_WINDOWS) + 5

class UnitStateBuffer:
    def __init__(self, maxlen: int = BUFFER_LEN):
        self.buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def push_and_featurize(self, unit_id: str, reading: dict, cycle: int) -> pd.DataFrame:
        row = {"unit_number": unit_id, "time, in cycles": cycle}
        row.update({c: reading.get(c, 0.0) for c in SETTINGS_COLS})
        row.update({c: reading.get(c, 0.0) for c in SENSOR_COLS})
        self.buffers[unit_id].append(row)

        hist = pd.DataFrame(list(self.buffers[unit_id]))
        feat_row = {}
        for col in ACTIVE_SENSORS:
            series = hist[col]
            for w in ROLLING_WINDOWS:
                feat_row[f"{col}_rmean_{w}"] = series.rolling(w, min_periods=1).mean().iloc[-1]
                feat_row[f"{col}_rstd_{w}"] = series.rolling(w, min_periods=1).std().fillna(0.0).iloc[-1]
            feat_row[f"{col}_ewma"] = series.ewm(span=10, adjust=False).mean().iloc[-1]
            feat_row[f"{col}_diff1"] = series.diff().fillna(0.0).iloc[-1]
        for col in SETTINGS_COLS + ACTIVE_SENSORS:
            feat_row[col] = hist[col].iloc[-1]
        return pd.DataFrame([feat_row])


class PredictiveMaintenanceEngine:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.classifier: FailureClassifier | None = None
        self.autoencoder: TorchAutoencoderAnomalyDetector | None = None
        self.scaler = None
        self.feature_cols: list[str] = []
        self.config: dict = {}
        self.buffer = UnitStateBuffer()
        self.loaded = False

    def load(self):
        import joblib
        self.classifier = FailureClassifier.load(os.path.join(self.model_dir, "xgb_classifier.joblib"))
        self.autoencoder = TorchAutoencoderAnomalyDetector.load(os.path.join(self.model_dir, "autoencoder.pt"))
        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))
        with open(os.path.join(self.model_dir, "feature_columns.json")) as f:
            self.feature_cols = json.load(f)
        with open(os.path.join(self.model_dir, "config.json")) as f:
            self.config = json.load(f)
        print(f"Loaded models from {self.model_dir}. Classifier threshold={self.config.get('decision_threshold', 0.5)}")
        self.loaded = True
        return self
    
    def score_reading(self, unit_id: str, reading: dict, cycle: int) -> dict:
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load() first, or train models via src/train.py")

        feat_df = self.buffer.push_and_featurize(unit_id, reading, cycle)
        for col in self.feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X = self.scaler.transform(feat_df[self.feature_cols].fillna(0.0))

        failure_prob = float(self.classifier.predict_prob(X)[0])
        threshold = self.config.get("decision_threshold", 0.5)
        failure_flag = int(failure_prob >= threshold)

        recon_error = float(self.autoencoder.score(X)[0])
        anomaly_flag = int(recon_error > self.autoencoder.threshold_)
        anomaly_severity = self.autoencoder.anomaly_ratio(recon_error)

        risk_score = float(np.clip(0.6 * failure_prob + 0.4 * min(anomaly_severity, 2.0) / 2.0, 0, 1))
        if risk_score >= 0.66 or failure_flag:
            status = "critical"
        elif risk_score >= 0.33 or anomaly_flag:
            status = "warning"
        else:
            status = "healthy"

        return {
            "unit_id": unit_id,
            "cycle": cycle,
            "failure_probbility": round(failure_prob, 4),
            "failure_predicted": bool(failure_flag),
            "anomaly_score": round(recon_error, 4),
            "anomaly_detected": bool(anomaly_flag),
            "anomaly_severity": round(anomaly_severity, 3),
            "risk_score": round(risk_score, 4),
            "status": status,
            "decision_threshold": threshold,
        }


# Global inference engine instance
# engine = PredictiveMaintenanceEngine()