import numpy as np
import pytest

from src.models.xgb_classifier import FailureClassifier
from src.models.autoencoder import AutoencoderAnomalyDetector
from src.evaluate import classification_metrics, best_threshold_by_fbeta


def make_separable_classification_data(n=400, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, n_features))
    weights = rng.normal(0, 1, size=n_features)
    logits = X @ weights
    y = (logits > np.median(logits)).astype(int)
    return X, y


def test_xgb_classifier_fits_and_predicts_proba_in_range():
    X, y = make_separable_classification_data()
    clf = FailureClassifier(n_estimators=50, max_depth=3)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_xgb_classifier_learns_signal_above_chance():
    X, y = make_separable_classification_data(n=600)
    split = 400
    clf = FailureClassifier(n_estimators=100, max_depth=4)
    clf.fit(X[:split], y[:split])
    preds = clf.predict(X[split:])
    acc = (preds == y[split:]).mean()
    assert acc > 0.7, f"expected classifier to beat chance comfortably, got acc={acc}"


def test_xgb_classifier_save_load_roundtrip(tmp_path):
    X, y = make_separable_classification_data()
    clf = FailureClassifier(n_estimators=30)
    clf.fit(X, y)
    path = tmp_path / "clf.joblib"
    clf.save(str(path))
    loaded = FailureClassifier.load(str(path))
    np.testing.assert_allclose(clf.predict_proba(X), loaded.predict_proba(X))


def test_autoencoder_flags_out_of_distribution_samples():
    rng = np.random.default_rng(1)
    X_healthy = rng.normal(0, 1, size=(300, 8))
    ae = AutoencoderAnomalyDetector(hidden_layers=(16, 4, 16), max_iter=300, threshold_percentile=95)
    ae.fit(X_healthy)

    # anomalous samples: shifted far from the healthy distribution
    X_anomalous = rng.normal(8, 1, size=(50, 8))
    flags = ae.predict_anomaly(X_anomalous)
    assert flags.mean() > 0.8, "most far-shifted samples should be flagged anomalous"

    # in-distribution samples should mostly NOT be flagged (threshold calibrated at 95th pct)
    X_normal_holdout = rng.normal(0, 1, size=(200, 8))
    flags_normal = ae.predict_anomaly(X_normal_holdout)
    assert flags_normal.mean() < 0.35


def test_autoencoder_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(100, 6))
    ae = AutoencoderAnomalyDetector(hidden_layers=(8, 3, 8), max_iter=200)
    ae.fit(X)
    path = tmp_path / "ae.joblib"
    ae.save(str(path))
    loaded = AutoencoderAnomalyDetector.load(str(path))
    assert loaded.threshold_ == ae.threshold_
    np.testing.assert_allclose(ae.score(X), loaded.score(X))


def test_classification_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.95])
    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["false_negatives"] == 0


def test_best_threshold_by_fbeta_returns_valid_probability():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=200)
    y_proba = np.clip(y_true * 0.5 + rng.normal(0, 0.3, size=200), 0, 1)
    t = best_threshold_by_fbeta(y_true, y_proba, beta=2.0)
    assert 0.0 <= t <= 1.0
