from __future__ import annotations
import argparse
import json
import os

import mlflow
import numpy as np
import optuna
from sklearn.model_selection import train_test_split

from src.feature_engineering import build_feature_matrix, get_feature_columns, fit_scaler, transform
from src.models.xgb_classifier import FailureClassifier
from src.models.autoencoder import TorchAutoencoderAnomalyDetector

from src.evaluate import (
    classification_metrics, anomaly_metrics, plot_roc_curve, plot_pr_curve,
    plot_confusion_matrix, save_metrics_json, best_threshold_by_fbeta,
)

from src.preprocessing import (
    load_cmapss, add_rul, add_failure_label, clip_rul,
    drop_low_variance_sensors, train_val_split_by_unit,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def prepare_data(data_dir: str, warning_window: int, val_fraction: float, seed: int):
    train_raw, test_raw, rul_truth = load_cmapss(data_dir, subset="FD001")

    train_raw = add_rul(train_raw)
    train_raw = clip_rul(train_raw, cap=130)
    train_raw = add_failure_label(train_raw, warning_window=warning_window)
    train_raw = drop_low_variance_sensors(train_raw)

    train_units, val_units = train_val_split_by_unit(train_raw, val_fraction, seed)

    train_feat = build_feature_matrix(train_units, fast=True)
    val_feat = build_feature_matrix(val_units, fast=True)

    feature_cols = get_feature_columns(train_feat)
    scaler = fit_scaler(train_feat, feature_cols)

    X_train = transform(train_feat, feature_cols, scaler)
    y_train = train_feat["failure_within_window"].to_numpy()
    X_val = transform(val_feat, feature_cols, scaler)
    y_val = val_feat["failure_within_window"].to_numpy()

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "feature_cols": feature_cols, "scaler": scaler,
        "train_feat": train_feat, "val_feat": val_feat,
    }


def objective(trial: optuna.Trial, data: dict) -> float:
    imbalance_ratio = (data["y_train"] == 0).sum() / max(1, (data["y_train"] == 1).sum())
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 1.0, imbalance_ratio * 1.5, log=True
        ),
    }
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        clf = FailureClassifier(**params)
        clf.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        proba = clf.predict_proba(data["X_val"])
        preds = (proba >= 0.5).astype(int)
        m = classification_metrics(data["y_val"], preds, proba)
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})

    trial.set_user_attr("pr_auc", m["pr_auc"])
    return m["pr_auc"] if not np.isnan(m["pr_auc"]) else 0.0


def run_autoencoder_training(data: dict, warning_window: int, out_dir: str):
    train_feat = data["train_feat"]
    val_feat = data["val_feat"]
    feature_cols = data["feature_cols"]

    # "Healthy" = RUL comfortably above the warning window -> early/mid life.
    healthy_mask = train_feat["RUL"] > (warning_window * 2)
    X_healthy = transform(train_feat[healthy_mask], feature_cols, data["scaler"])

    ae = TorchAutoencoderAnomalyDetector(hidden_dims=(64, 32, 8), max_epochs=200,
                                          patience=15, threshold_percentile=99.0)
    ae.fit(X_healthy)

    X_val = data["X_val"]
    y_val = data["y_val"]
    scores = ae.score(X_val)
    flags = ae.predict_anomaly(X_val)
    m = anomaly_metrics(y_val, flags, scores)
    m["epochs_trained"] = len(ae.train_losses_)
    m["final_train_mse"] = ae.train_losses_[-1]
    m["final_val_mse"] = ae.val_losses_[-1]

    ae.save(os.path.join(out_dir, "autoencoder.pt"))
    return ae, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    
    ap.add_argument("--out-dir", default="./models")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--warning-window", type=int, default=30)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--experiment", default="turbofan-failure-prediction")
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    mlflow.set_tracking_uri(args.mlflow.uri)
    mlflow.set_experiment(args.experiment)
    
    print("Loading + engineering features...")
    data = prepare_data(args.data_dir, args.warning_window, args.val_fraction, args.seed)
    print(f"Train rows: {len(data['y_train'])} | Val rows: {len(data['y_val'])} "
          f"| Features: {len(data['feature_cols'])}")
    
    with mlflow.start_run(run_name="xgb_optuna_search") as parent_run:
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("warning_window", args.warning_window)

        print(f"Running Optuna search ({args.n_trials} trials)...")
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(lambda t: objective(t, data), n_trials=args.n_trials, show_progress_bar=False)

        best_params = study.best_params
        print(f"Best params: {best_params}")
        print(f"Best val PR-AUC: {study.best_value:.4f}")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_pr_auc", study.best_value)

        # Retrain best model
        best_clf = FailureClassifier(**best_params)
        best_clf.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        val_proba = best_clf.predict_proba(data["X_val"])
        decision_threshold = best_threshold_by_fbeta(data["y_val"], val_proba, beta=2.0)
        val_preds = (val_proba >= decision_threshold).astype(int)
        final_metrics = classification_metrics(data["y_val"], val_preds, val_proba)
        final_metrics["decision_threshold"] = decision_threshold
        mlflow.log_metric("decision_threshold", decision_threshold)
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()
                             if isinstance(v, (int, float)) and not np.isnan(v)})

        os.makedirs("artifacts", exist_ok=True)
        plot_roc_curve(data["y_val"], val_proba, "artifacts/roc_curve.png")
        plot_pr_curve(data["y_val"], val_proba, "artifacts/pr_curve.png")
        plot_confusion_matrix(data["y_val"], val_preds, "artifacts/confusion_matrix.png")
        save_metrics_json(final_metrics, "artifacts/classifier_metrics.json")
        mlflow.log_artifacts("artifacts")

        best_clf.save(os.path.join(args.out_dir, "xgb_classifier.joblib"))

        print("Training autoencoder anomaly detector...")
        ae, ae_metrics = run_autoencoder_training(data, args.warning_window, args.out_dir)
        mlflow.log_metrics({f"ae_{k}": v for k, v in ae_metrics.items()
                             if isinstance(v, (int, float)) and not np.isnan(v)})
        save_metrics_json(ae_metrics, "artifacts/autoencoder_metrics.json")
        mlflow.log_artifact("artifacts/autoencoder_metrics.json")

        # Persist scaler + feature schema for the API
        import joblib
        joblib.dump(data["scaler"], os.path.join(args.out_dir, "scaler.joblib"))
        with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
            json.dump(data["feature_cols"], f)
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump({"warning_window": args.warning_window,
                       "best_params": best_params,
                       "decision_threshold": decision_threshold,
                       "mlflow_run_id": parent_run.info.run_id}, f, indent=2)

    print("\n=== Final classifier metrics (validation) ===")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")
    print("\n=== Autoencoder metrics (validation, unsupervised) ===")
    for k, v in ae_metrics.items():
        print(f"  {k}: {v}")
    print(f"\nModels saved to {args.out_dir}/")
    print(f"MLflow tracking data at {args.mlflow_uri} -- run `mlflow ui --backend-store-uri {args.mlflow_uri}`")


if __name__ == "__main__":
    main()
    