# Implementation Guide

Detailed documentation of entrypoints, setup procedures, execution steps, and architectural decisions made throughout the project.

---

## 1. Environment Setup

### Python Environment

- **Python version**: 3.12.3
- **Virtual environment**: `venv` (created via `python3 -m venv venv`)
- **Location**: `/home/aryan/Desktop/ride-price-prediction/venv/`

### Dependencies

All dependencies are listed in `requirements.txt`. Key libraries and their roles:

| Library | Version | Purpose |
|---|---|---|
| numpy | 2.5.2 | Numerical operations |
| pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.9.0 | ML models, preprocessing, metrics |
| xgboost | 3.4.1 | Gradient boosting (best model) |
| lightgbm | 4.7.0 | Gradient boosting (alternative) |
| mlflow | 3.15.2 | Experiment tracking & model registry |
| fastapi | 0.141.1 | REST API framework |
| uvicorn | 0.52.4 | ASGI server |
| evidently | 0.7.21 | Data drift detection |
| dvc | 3.67.1 | Data version control |
| pytest | 9.1.1 | Testing framework |

### Version Control

- **Git**: Initialized with 5 commits tracking each phase
- **DVC**: Initialized for data tracking with local cache (Google Drive remote configurable)

---

## 2. Entrypoints

Every module can be run as a standalone script via `python -m <module_path>`.

### Data Pipeline

| Command | What it does | Input | Output |
|---|---|---|---|
| `python -m src.data.generate` | Generates 100K synthetic ride records | configs/config.yaml | data/raw/rides.csv |
| `python -m src.data.preprocess` | Cleans, encodes, scales, splits data | data/raw/rides.csv | data/features/train.csv, data/features/test.csv, models/scaler.pkl, models/encoder.pkl |
| `python -m src.features.build_features` | Engineers derived features | data/features/train.csv, data/features/test.csv | Updated train.csv, test.csv (22 columns) |

### Model Training

| Command | What it does | Input | Output |
|---|---|---|---|
| `python -m src.models.train` | Trains 5 models, logs to MLflow, saves best | data/features/train.csv, data/features/test.csv | models/best_model.pkl, reports/metrics.json, reports/model_comparison.csv |

### API

| Command | What it does | Port |
|---|---|---|
| `python -m src.api.main` | Starts FastAPI server | 8000 |

### Monitoring

| Command | What it does | Output |
|---|---|---|
| `python -m src.monitoring.drift` | Runs drift detection on current data | reports/drift/*.html, reports/drift/*.json |
| `python -m src.monitoring.simulate_drift` | Runs 5 drift scenarios | reports/drift/simulation_results.json, reports/drift/*.html |

### Testing

| Command | What it does |
|---|---|
| `python -m pytest tests/ -v` | Runs all 60 tests |
| `python -m pytest tests/test_data.py -v` | Data tests only (28 tests) |
| `python -m pytest tests/test_model.py -v` | Model tests only (11 tests) |
| `python -m pytest tests/test_api.py -v` | API tests only (21 tests) |

### MLflow UI

| Command | What it does | Port |
|---|---|---|
| `mlflow ui --port 5000` | Opens MLflow experiment tracker | 5000 |

---

## 3. Configuration

### configs/config.yaml

Central configuration file containing:

- **paths**: All data/model/report file paths
- **data**: n_samples (100K), random_seed (42), test_size (0.2), target_column
- **features**: Lists of numeric and categorical features
- **model.experiments**: 5 model configs with hyperparameters
- **mlflow**: Experiment name, tracking URI, model registry name
- **api**: Host, port, title, version
- **monitoring**: Drift thresholds

### params.yaml

DVC pipeline parameters (mirrors config.yaml for DVC tracking).

### dvc.yaml

Defines the DVC pipeline stages: `generate` -> `preprocess` -> `build_features` -> `train`.

---

## 4. Architectural Decisions

### Decision 1: Synthetic Data with Deterministic Target

**Choice**: Generated the `price_multiplier` target using a formula based on demand/supply ratio, weather severity, and time patterns, rather than random labels.

**Rationale**: This creates a realistic signal-to-noise ratio where models can learn meaningful patterns. The target formula includes:
- 30% weight on demand/supply ratio
- 8% weight on weather severity
- 15% rush hour premium
- 5% weekend premium
- Gaussian noise (std=0.05)

This produces models with R2 ~0.99 on tree-based models and ~0.89 on linear models, demonstrating the value of non-linear approaches.

### Decision 2: Feature Engineering Before Encoding

**Choice**: Applied feature engineering (demand_supply_ratio, is_rush_hour, weather_severity_score, etc.) after preprocessing but before model training, creating 22 total features.

**Rationale**: Features like `demand_supply_ratio` and `demand_weather_interaction` capture domain-specific interactions that linear models would struggle to learn on their own.

### Decision 3: XGBoost as Best Model

**Choice**: Selected XGBoost based on test RMSE (0.0478) and fast training time (7s).

**Rationale**: XGBoost provides the best balance of accuracy and training speed. LightGBM is close in performance (4s training) and could be preferred at larger scales.

### Decision 4: Eager Model Loading in API

**Choice**: Used a singleton pattern with `get_predictor()` for lazy-loading the model on first request, rather than loading at import time.

**Rationale**: This avoids import-time side effects, makes testing easier (via `reset_predictor()`), and ensures the API starts quickly. The model loads on the first request and stays in memory.

### Decision 5: Separate Preprocessing Artifacts

**Choice**: Saved `scaler.pkl` and `encoder.pkl` alongside the model, and stored feature names in the XGBoost model itself.

**Rationale**: The prediction pipeline needs consistent preprocessing. By saving these artifacts and using the model's feature names for ordering, we prevent train/serve skew.

### Decision 6: Evidently AI 0.7.x API

**Choice**: Adapted to Evidently 0.7.21 which uses `Report` + `dump_dict()` instead of the older `Report.as_dict()` API.

**Rationale**: The installed version (0.7.21) uses a completely different API from earlier versions. The drift detector parses the `metric_results` dict from `dump_dict()` and extracts per-column drift status from widget counter labels.

### Decision 7: Comprehensive Test Suite (60 tests)

**Choice**: Wrote 60 tests covering data generation (15), preprocessing (8), model prediction (11), evaluation (4), and API endpoints (21).

**Rationale**: Comprehensive testing ensures each pipeline stage works independently and the API correctly validates inputs, handles errors, and returns expected schemas.

### Decision 8: Pydantic v2 for API Validation

**Choice**: Used Pydantic v2 `BaseModel` with `model_dump()` for request/response validation.

**Rationale**: Pydantic v2 provides automatic JSON Schema generation, strict type validation, and integrates natively with FastAPI's OpenAPI docs.

---

## 5. Data Flow

```
1. generate.py
   └── Creates 100K rows with 12 columns
   └── Target (price_multiplier) derived from formula

2. preprocess.py
   └── Drops ride_id, timestamp
   └── One-hot encodes weather_condition (5 categories -> 5 columns)
   └── StandardScaler on all numeric features
   └── 80/20 train/test split
   └── Saves scaler.pkl, encoder.pkl

3. build_features.py
   └── demand_supply_ratio = demand / (drivers + 1)
   └── is_rush_hour (7-9 AM, 5-7 PM)
   └── weather_severity_score (0-4 ordinal)
   └── hour_sin, hour_cos (cyclical encoding)
   └── is_night (10 PM - 5 AM)
   └── demand_weather_interaction
   └── fare_per_driver
   └── Final: 22 features

4. train.py
   └── Trains 5 models with MLflow tracking
   └── Logs params, metrics, model artifacts
   └── Saves best model to models/best_model.pkl
   └── Saves comparison to reports/model_comparison.csv

5. API (main.py)
   └── Loads model + scaler + encoder on first request
   └── /predict: preprocess -> engineer features -> encode -> predict
   └── Returns multiplier, estimated price, confidence

6. Drift Detection (drift.py)
   └── Loads training data as reference
   └── Compares against current data using Evidently
   └── Generates HTML report + JSON summary
   └── Reports per-column drift scores
```

---

## 6. Running the Complete Pipeline

### From Scratch

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Full pipeline
python -m src.data.generate
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train

# Verify
python -m pytest tests/ -v
python -m src.models.predict

# Deploy
python -m src.api.main
# or
docker compose up --build
```

### With DVC

```bash
dvc repro  # Runs generate -> preprocess -> build_features -> train
```

### Monitoring

```bash
python -m src.monitoring.simulate_drift  # 5 drift scenarios
```
