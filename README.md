
## Dataset

Built against the schema of NASA's **C-MAPSS Turbofan Engine Degradation
Simulation** dataset (`unit_number, time_in_cycles, op_setting_1..3,
sensor_1..21`). This sandbox had no network access to `data.nasa.gov`, so
`data/generate_synthetic_cmapss.py` produces synthetic run-to-failure data
with the same schema, degradation trends, and noise characteristics. **To
use the real dataset**, download `train_FD001.txt`, `test_FD001.txt`, and
`RUL_FD001.txt` from the [NASA Prognostics Data
Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
and drop them into `./data/` — no code changes needed, `src/preprocessing.py`
reads the exact same format.

## Project layout

```
data/         real CMAPSS .txt files
src/
  preprocessing.py       RUL labeling, failure-window labeling, leak-safe splits
  feature_engineering.py rolling mean/std, EWMA, diffs, degradation slope
  models/
    xgb_classifier.py    XGBoost binary failure classifier
    autoencoder.py       autoencoder anomaly detector
  evaluate.py             metrics (accuracy/precision/recall/F1/ROC-AUC/PR-AUC/
                           MCC/Brier), plots, F-beta optimal threshold search
  train.py                MLflow-tracked training pipeline + Optuna HPO
api/
  main.py          FastAPI app: REST endpoints, WebSocket live feed, serves dashboard
  inference.py     Loads trained models, streaming feature engineering, scoring
  schemas.py       Pydantic request/response models
dashboard/         Static HTML/CSS/JS control-room dashboard (Chart.js + WebSocket)
docker/            Dockerfile.api, Dockerfile.train, docker-compose.yml
tests/             pytest suite: preprocessing, models, API (21 tests)
models/            serialized model artifacts (created by training)
artifacts/         evaluation plots + metrics JSON (created by training)
```



## How the ML pipeline works

**Labels.** Two targets are derived from each unit's run-to-failure history:
`RUL` (remaining useful life, cycles until failure, capped at 130 per the
standard CMAPSS piecewise-linear convention) and `failure_within_window`
(binary: will this unit fail within the next 30 cycles — the classifier's
actual target).

**Features.** Per-sensor rolling mean/std (windows of 5 and 15 cycles),
exponentially-weighted moving average, and first difference, computed
causally per unit (no look-ahead). 21 raw sensors → ~100 engineered
features after dropping 7 near-constant sensors.

**XGBoost classifier.** Binary classifier for "failure within warning
window." Class imbalance is handled via a tunable `scale_pos_weight`.
Optuna runs a TPE search (default 20-25 trials) over tree depth, learning
rate, subsampling, and regularization, optimizing validation PR-AUC (robust
to imbalance, unlike accuracy). Every trial is logged to MLflow as a nested
run; the best configuration is retrained on the full training set. The
decision threshold is then chosen to maximize **F2** (recall weighted 2x
over precision) on validation — because in maintenance, a missed failure
(false negative) is far more costly than a false alarm.

**Autoencoder anomaly detector.** Trained only on "healthy" cycles (RUL well
above the warning window) to reconstruct its own input; reconstruction
error becomes an anomaly score. Flags equipment drifting into an
unfamiliar operating regime *without* needing labeled failures — useful for
catching novel fault modes the classifier was never trained on. Implemented
with a symmetric-bottleneck `MLPRegressor` (64-32-8-32-64) rather than
PyTorch/TensorFlow, since this sandbox's disk quota couldn't fit either
framework; swap in a real `nn.Module` in `src/models/autoencoder.py` if you
have the disk budget — the rest of the pipeline (fit/score/save/load,
threshold logic) is framework-agnostic.

**Evaluation.** `src/evaluate.py` computes accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, Matthews correlation coefficient, Brier score, and full
confusion-matrix breakdown (TP/TN/FP/FN + false-negative rate) for both
models, plus ROC/PR-curve and confusion-matrix plots logged as MLflow
artifacts. On the synthetic dataset: classifier ROC-AUC ≈ 0.97, F1 ≈ 0.67,
recall ≈ 0.92; autoencoder ROC-AUC ≈ 0.96 (see `artifacts/*.json` after
training, or `GET /api/metrics`).

**Experiment tracking.** All runs (parent + nested Optuna trials) are
logged to MLflow with params, metrics, and artifacts. Backend is SQLite
(`sqlite:///mlflow.db`) since the newer MLflow versions deprecated the
plain-filesystem store.

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness + model-load status |
| POST | `/api/predict` | Score one sensor reading |
| POST | `/api/predict/batch` | Score a list of readings |
| GET | `/api/fleet/summary` | Aggregate healthy/warning/critical counts + avg risk |
| GET | `/api/fleet/units` | Current state of every simulated unit |
| GET | `/api/fleet/history/{unit_id}` | Recent time series for one unit |
| GET | `/api/metrics` | Latest offline classifier/autoencoder evaluation metrics |
| WS | `/ws/live` | Push feed: a `snapshot` message on connect, then a `tick` message every `SIM_TICK_SECONDS` |

Example:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"unit_id": "turbine-07", "cycle": 210, "sensor_2": 560, "sensor_3": 650}'
```

## Real-time dashboard

Served at `/`. Connects to `/ws/live`, falls back to polling
`/api/fleet/units` every 2s if the socket drops (auto-reconnects every 3s).
Shows fleet-wide KPIs, a gauge card per equipment unit (radial dial +
status pill), a live trend chart (risk score / failure probability) for
whichever unit you select, and the latest offline model-evaluation metrics.

