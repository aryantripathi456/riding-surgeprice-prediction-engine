# Dynamic Pricing Engine

An end-to-end MLOps project that predicts optimal ride price multipliers based on real-time supply, demand, and weather conditions.

## Overview

This project implements a production-ready dynamic pricing engine for ride-sharing platforms. It covers the full MLOps lifecycle: synthetic data generation, feature engineering, model training with experiment tracking, API deployment with containerization, and continuous monitoring with drift detection.

## Tech Stack

| Category | Tools |
|---|---|
| Data Engineering | Python, Pandas, NumPy |
| Data Versioning | DVC |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Experiment Tracking | MLflow |
| Deployment | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Monitoring | Evidently AI |
| Testing | Pytest |

## Project Structure

```
ride-price-prediction/
├── configs/
│   └── config.yaml           # Central configuration
├── data/
│   ├── raw/                   # DVC-tracked raw data
│   ├── processed/             # Cleaned data
│   └── features/              # Train/test splits with engineered features
├── models/
│   ├── best_model.pkl         # Best model (XGBoost)
│   ├── scaler.pkl             # Fitted StandardScaler
│   └── encoder.pkl            # Fitted OneHotEncoder
├── src/
│   ├── data/
│   │   ├── generate.py        # Synthetic data generation (100K rows)
│   │   └── preprocess.py      # Cleaning, encoding, scaling
│   ├── features/
│   │   └── build_features.py  # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py           # Model training + MLflow tracking
│   │   └── predict.py         # Inference wrapper
│   ├── api/
│   │   ├── main.py            # FastAPI application
│   │   └── schemas.py         # Pydantic request/response models
│   ├── monitoring/
│   │   ├── drift.py           # Evidently AI drift detection
│   │   └── simulate_drift.py  # Drift simulation demo
│   └── utils.py               # Shared utilities
├── tests/
│   ├── test_data.py           # Data generation & preprocessing tests
│   ├── test_model.py          # Model & prediction tests
│   └── test_api.py            # API endpoint tests
├── reports/
│   └── drift/                 # Evidently drift reports (HTML + JSON)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                   # DVC pipeline definition
├── params.yaml                # DVC pipeline parameters
└── Dynamic_Pricing_Engine_Project.docx
```

## Model Performance

| Model | Test RMSE | Test R2 | Training Time |
|---|---|---|---|
| **XGBoost** | **0.0478** | **0.9949** | 7.0s |
| LightGBM | 0.0482 | 0.9948 | 4.0s |
| Random Forest | 0.0521 | 0.9939 | 96.8s |
| Ridge Regression | 0.2245 | 0.8883 | 0.07s |
| Linear Regression | 0.2238 | 0.8890 | 0.39s |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict price multiplier |
| GET | `/model-info` | Model version and metrics |

### Example Request

```json
POST /predict
{
  "hour_of_day": 18,
  "day_of_week": 2,
  "is_weekend": false,
  "passenger_demand": 50,
  "driver_availability": 10,
  "weather_condition": "rain",
  "temperature": 12.5,
  "visibility_km": 5.0,
  "base_fare": 8.50
}
```

### Example Response

```json
{
  "price_multiplier": 2.83,
  "estimated_price": 24.04,
  "confidence": "low"
}
```

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (optional)

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd ride-price-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init
```

### Run the Full Pipeline

```bash
# Generate synthetic data (100K rows)
python -m src.data.generate

# Preprocess and split
python -m src.data.preprocess

# Engineer features
python -m src.features.build_features

# Train all models + MLflow tracking
python -m src.models.train
```

### Run the API

```bash
# Start the server
python -m src.api.main

# Visit Swagger docs at http://localhost:8000/docs
```

### Run with Docker

```bash
docker compose up --build
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Drift Simulation

```bash
python -m src.monitoring.simulate_drift
```

### View MLflow Experiments

```bash
mlflow ui --port 5000
# Visit http://localhost:5000
```

## DVC Pipeline

The project includes a complete DVC pipeline (`dvc.yaml`) with four stages:

```
generate -> preprocess -> build_features -> train
```

To reproduce the full pipeline:

```bash
dvc repro
```

To add Google Drive remote for data versioning:

```bash
dvc remote add -d gdrive gdrive://<your-folder-id>
dvc push
```

## License

This project is for educational purposes.
