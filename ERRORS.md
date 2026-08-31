# Errors Encountered and Fixes

A chronological record of every significant error encountered during development and the fix applied.

---

## Phase 0: Environment Setup

### Error 1: pip install blocked by PEP 668

**Command**: `pip install python-docx`

**Error**:
```
error: externally-managed-environment
note: See PEP 668 for more information.
```

**Cause**: Python 3.12 on Ubuntu uses an externally-managed-environment marker that prevents system-wide pip installs outside of a virtual environment.

**Fix**: Used `--break-system-packages` flag:
```bash
pip install python-docx --break-system-packages
```

---

## Phase 1: Data Engineering

### Error 2: DVC add fails due to git-ignored .dvc files

**Command**: `dvc add data/raw/rides.csv`

**Error**:
```
ERROR: bad DVC file name 'data/raw/rides.csv.dvc' is git-ignored.
```

**Cause**: The `.gitignore` had `/data/raw/` which git interpreted as ignoring everything under that directory, including `data/raw/rides.csv.dvc`.

**Fix**: Changed the gitignore patterns to ignore directory contents but allow `.dvc` tracking files:
```gitignore
# Before (broken)
/data/raw/

# After (fixed)
/data/raw/*
!/data/raw/*.dvc
```

---

## Phase 2: Model Training

### Error 3: MLflow XGBoost logging fails with skops trust error

**Command**: `python -m src.models.train`

**Error**:
```
mlflow.exceptions.MlflowException: The saved sklearn model references untrusted types.
Root error: Untrusted types found in the file:
['xgboost.core.Booster', 'xgboost.sklearn.XGBRegressor'].
```

**Cause**: MLflow 3.x uses `skops` for model serialization which requires explicit trust for non-sklearn types like XGBoost. The default `mlflow.sklearn.log_model()` does not trust XGBoost types.

**Fix**: Used the correct MLflow flavor for each model type:
```python
# Before (broken)
mlflow.sklearn.log_model(model, artifact_path="model")

# After (fixed)
if model_class_name == "XGBRegressor":
    mlflow.xgboost.log_model(model, artifact_path="model")
elif model_class_name == "LGBMRegressor":
    mlflow.lightgbm.log_model(model, artifact_path="model")
else:
    mlflow.sklearn.log_model(model, artifact_path="model")
```

### Error 4: Training timeout (5 minutes)

**Command**: `python -m src.models.train`

**Error**: Process killed after 300s timeout.

**Cause**: Default hyperparameters (RF: 200 estimators, XGBoost/LightGBM: 300 estimators) on 80K rows were too slow for the available hardware. Random Forest alone took ~97 seconds.

**Fix**: Reduced model complexity in `configs/config.yaml`:
```yaml
# Before
n_estimators: 200  # RF
n_estimators: 300  # XGBoost, LightGBM

# After
n_estimators: 100  # RF
n_estimators: 150  # XGBoost, LightGBM
```

---

## Phase 3: API & Deployment

### Error 5: Prediction fails with feature name mismatch

**Command**: `python -m src.models.predict`

**Error**:
```
ValueError: feature_names mismatch:
['hour_of_day', ..., 'weather_condition_clear', ..., 'fare_per_driver']
['hour_of_day', ..., 'fare_per_driver', ..., 'weather_condition_clear']
```

**Cause**: During training, columns were in the order they appeared after `get_dummies()` (weather columns at the end). During prediction, the `preprocess()` method built features in a different order. XGBoost is strict about feature name matching.

**Fix**: Stored the model's expected feature names and reordered columns before prediction:
```python
# In __init__
self.feature_names = list(self.model.get_booster().feature_names)

# In preprocess (at the end)
if self.feature_names:
    df = df[self.feature_names]
```

### Error 6: API tests return 503 "Model not loaded"

**Command**: `python -m pytest tests/test_api.py`

**Error**: All `/predict` tests failed with `503 Service Unavailable`.

**Cause**: The `TestClient(app)` fixture was not triggering the FastAPI lifespan context manager. Without the lifespan, the `predictor` global remained `None`. The model was only loaded inside the `lifespan` function.

**Fix**: Refactored the API to use a lazy singleton pattern instead of relying on lifespan:
```python
# Before: Model only loaded in lifespan
predictor = None
@asynccontextmanager
async def lifespan(app):
    global predictor
    predictor = PricePredictor(config=config)
    yield
    predictor = None

# After: Model loaded on first request via singleton
_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor(config=_config)
    return _predictor
```

Also added a `reset_predictor()` function for test isolation:
```python
@pytest.fixture(autouse=True)
def reset_model():
    reset_predictor()
    yield
    reset_predictor()
```

---

## Phase 4: Monitoring

### Error 7: ImportError - ColumnMapping not in evidently

**Command**: `python -m src.monitoring.drift`

**Error**:
```
ImportError: cannot import name 'ColumnMapping' from 'evidently'
```

**Cause**: Evidently 0.7.x completely restructured its API. `ColumnMapping`, `Report` from `evidently.report`, and `DataDriftPreset` from `evidently.metric_preset` no longer exist. The new API uses `evidently.Report`, `evidently.presets.DataDriftPreset`.

**Fix**: Rewrote the drift module using the new API:
```python
# Before (broken)
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# After (fixed)
from evidently import Report
from evidently.presets import DataDriftPreset
```

### Error 8: 'Snapshot' object has no attribute 'as_dict'

**Command**: `python -m src.monitoring.drift`

**Error**:
```
AttributeError: 'Snapshot' object has no attribute 'as_dict'
```

**Cause**: In Evidently 0.7.x, `Report.run()` returns a `Snapshot` object (not a `Report`). The `Snapshot` class has `dump_dict()` instead of `as_dict()`.

**Fix**: Changed to `dump_dict()`:
```python
# Before
result = report.as_dict()

# After
snapshot = report.run(reference, current)
result = snapshot.dump_dict()
```

### Error 9: AttributeError on widget params (NoneType)

**Command**: `python -m src.monitoring.drift`

**Error**:
```
AttributeError: 'NoneType' object has no attribute 'get'
```

**Cause**: Some widgets in the Evidently output have `params: null` instead of an empty dict. The code called `.get("counters", [])` on `None`.

**Fix**: Added null-safe access:
```python
# Before
counters = widget.get("params", {}).get("counters", [])

# After
params = widget.get("params") or {}
counters = params.get("counters", [])
```

### Error 10: NaN warnings during drift simulation

**Command**: `python -m src.monitoring.simulate_drift`

**Error**:
```
RuntimeWarning: invalid value encountered in divide
```

**Cause**: When multiplying `driver_availability` by 0.3 in severe scenarios, some values clipped to 0, causing division by zero in `fare_per_driver = base_fare / (drivers + 1)` when drivers reached 0 after noise addition.

**Fix**: Added `.clip(lower=1)` to driver availability after modification:
```python
scenario["driver_availability"] = (
    scenario["driver_availability"] * params["driver_reduction"]
    + rng.normal(0, 2, size=len(scenario))
).clip(lower=1).astype(int)
```

---

## Test Fixes

### Error 11: Median assertion wrong

**Test**: `test_fills_numeric_with_median`

**Error**: `assert result["a"].iloc[1] == 3.0` failed (got 3.5)

**Cause**: Median of [1, 3, 4, 5] is 3.5, not 3.0. The median is the average of the two middle values.

**Fix**: Changed assertion to `== 3.5`.

### Error 12: OneHotEncoder test expects unknown column

**Test**: `test_transform_uses_fitted_encoder`

**Error**: `assert "weather_snow" in result_test.columns` failed

**Cause**: `OneHotEncoder(handle_unknown="ignore")` drops unknown categories (all zeros for unknown), it doesn't create a new column for them.

**Fix**: Changed test to verify that unknown categories get zero encoding:
```python
# "snow" is unknown to encoder, gets all zeros
assert result_test["weather_clear"].iloc[1] == 0.0
```

### Error 13: StandardScaler std assertion too strict

**Test**: `test_scales_to_standard`

**Error**: `assert abs(result["a"].std() - 1.0) < 0.1` failed (got 1.118)

**Cause**: `pandas.Series.std()` uses sample std (ddof=1), while `StandardScaler` uses population std (ddof=0). For n=5, the sample std is sqrt(5/4) = 1.118, not 1.0.

**Fix**: Used `ddof=0` to match StandardScaler behavior:
```python
assert abs(result["a"].std(ddof=0) - 1.0) < 0.01
```

---

## Summary

| # | Phase | Error | Root Cause | Fix |
|---|---|---|---|---|
| 1 | Setup | PEP 668 blocked pip | System Python protection | `--break-system-packages` |
| 2 | DVC | .dvc file git-ignored | Overly broad gitignore pattern | Negation pattern `!*.dvc` |
| 3 | Training | MLflow skops trust | XGBoost types not trusted | Use `mlflow.xgboost.log_model` |
| 4 | Training | 5-minute timeout | Too many estimators | Reduced n_estimators |
| 5 | Prediction | Feature name mismatch | Column order inconsistency | Reorder by model feature names |
| 6 | API | 503 Model not loaded | Lifespan not triggered by TestClient | Lazy singleton + reset fixture |
| 7 | Monitoring | Import error | Evidently 0.7.x API change | New imports from `evidently.presets` |
| 8 | Monitoring | No as_dict method | Snapshot vs Report API | Use `dump_dict()` |
| 9 | Monitoring | NoneType get error | Null widget params | `widget.get("params") or {}` |
| 10 | Monitoring | Division by zero | Driver count clipping | `.clip(lower=1)` |
| 11 | Tests | Wrong median | Arithmetic error | Fixed to 3.5 |
| 12 | Tests | Encoder ignores unknown | OneHotEncoder behavior | Updated assertion |
| 13 | Tests | Std mismatch | ddof=0 vs ddof=1 | Use `std(ddof=0)` |
