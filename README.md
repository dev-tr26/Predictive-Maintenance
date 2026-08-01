## Dataset
 
- Data Link : [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) 
- [Kaggle-link](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data)
- time-series sensor telemetry from equipment

```
Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX 
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX 
Fault Modes: TWO (HPC Degradation, Fan Degradation)
```

------

### problem statement - predict when an engine is about to fail 

1. Supervised binary classification: prob that unit fail within the next N cycles 
2. Unsupervised anomaly detection: catches failure modes the classifier was never trained on

- Both run on every incoming reading; their outputs are blended into a single risk score.

------

## Preprocessing (preprocessing.py)
      
### 1. RUL(remaining useful life) for each unit capping at 130 
- the standard trick from the original CMAPSS literature: it turns the target into a piecewise-linear curve (flat plateau, then linear decay) so the model concentrates on the degradation-visible region


### 2. Binary failure label 
- Why not pure RUL regression But classification ? 
- maintaenance is done on thresholds like (service next month) classification maps directly to that prob. 
- picked a 30-cycle warning window 


### 3. Dropping low-variance sensors
- sensors 1, 5, 6, 10, 16, 18, 19 are dropped bca they are functionally constant. 
- just add dimensionality without signal 


### 4. Train/val split by unit, not by row
- splitting by row would put cycle 50 of unit-7 in train and cycle 51 of unit-7 in validation
- the model would essentially memorize each unit's trajectory and validation performance would be a lie.
- Splitting by whole unit_number (20% of units held out entirely) is leak-safe approach for this kind of longitudinal data.

------


## Feature engineering (feature_engineering.py)

- Raw sensor values = noisy single-point readings 
- a spike doesn't tell you if it's noise or the start of degradation 
- per sensor per unit computes
- turns 21 raw sensors into ~100 features (14 active sensors × ~7 derived features each)

#### 1. Rolling mean/std over 5- and 15-cycle windows :  smooths noise, captures the level and volatility of recent behavior
#### 2. EWMA (span=10) : recency-weighted trend, reacts faster than a plain rolling mean to a genuine shift while still damping single-point noise
#### 3. First difference :  instantaneous rate of change, catches sudden jumps a rolling average would smear out


------

## Model choice

### 1. XGBoost for classifier : 
- bcz tabular,structured moderate-size data , fast to train , has scale_pos_weight for class imbalance (failure-window rows are a small minority)

### 2. Autoencoder for anomaly detection : 

- xgb cannot detect failure mode if it never saw a labelled eg. of it , autoencoder is trained only on healthy cycles (RUL > warning_window × 2 )
- A degrading engine drifts into sensor combinations the network never learned to reconstruct well, so reconstruction error spikes -> no failure labels required 
- symmetric bottleneck 64→32→8→32→64, ReLU + dropout, Adam optimizer, MSE reconstruction loss, early stopping on a held-out healthy validation slice.

------


## Evaluation metrics (src/evaluate.py)

- Accuracy : useless bcz class imbalanced 
- PR-AUC : more informative than ROC-AUC under class imbalance , ROC-AUC can look deceptively good even when precision on the minority (failure) class is poor.
- FNR : a missed failure (false negative) is the costly error (safety incidents and unplanned downtime )
- MCC : MCC is a single balanced summary robust to imbalance
- Brier score : checks whether the predicted probabilities are well-calibrated (not just the binary decision).


----- 


### Decision Threshold

- default 0.5 classification threshold on the first training run, gave recall of 0.05 — the model was almost never flagging real failures because 0.5 doesn't account for the class imbalance
- best_threshold_by_fbeta(beta=2.0) : picks threshold that maximizes F2 (recall weighted twice as heavily as precision) on val set. 


-----

## Optuna hyperparameter search

```
n_estimators:      100–500, step 50     # number of trees
max_depth:         3–9                  # tree depth
learning_rate:     0.01–0.3, log scale  # shrinkage per tree
subsample:         0.6–1.0              # row sampling per tree
colsample_bytree:  0.6–1.0              # feature sampling per tree
min_child_weight:  1–10                 # min samples per leaf
reg_lambda:        1e-3–10, log scale   # L2 regularization
reg_alpha:         1e-3–10, log scale   # L1 regularization
scale_pos_weight:  1.0 – (imbalance_ratio × 1.5), log scale
```



------


### Final retrain + serialization

```
Best Optuna params -> retrain on the full training set -> evaluate on held-out validation with the F2-tuned threshold -> save xgb_classifier.joblib, autoencoder.pt, scaler.joblib, feature_columns.json, and config.json
```

- done this for FD001 | FD002 | FD003 | FD004 different datasets 

- All runs (parent + nested Optuna trials) are
logged to MLflow with params, metrics, and artifacts. Backend is SQLite
(`sqlite:///mlflow.db`)

-----


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



## Real-time dashboard

Served at `/`. Connects to `/ws/live`, falls back to polling
`/api/fleet/units` every 2s if the socket drops (auto-reconnects every 3s).
Shows fleet-wide KPIs, a gauge card per equipment unit (radial dial +
status pill), a live trend chart (risk score / failure probability) for
whichever unit you select, and the latest offline model-evaluation metrics.

