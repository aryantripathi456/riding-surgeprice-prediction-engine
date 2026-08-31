"""Comprehensive tests for model training and prediction."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.predict import PricePredictor
from src.data.generate import generate_synthetic_data
from src.data.preprocess import handle_missing_values, drop_unnecessary_columns, encode_categoricals


@pytest.fixture
def sample_training_data():
    """Create small sample training data for fast tests."""
    df = generate_synthetic_data(n_samples=500, random_seed=42)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)

    # Add engineered features
    df["demand_supply_ratio"] = df["passenger_demand"] / (df["driver_availability"] + 1)
    df["is_rush_hour"] = (
        ((df["hour_of_day"] >= 7) & (df["hour_of_day"] <= 9))
        | ((df["hour_of_day"] >= 17) & (df["hour_of_day"] <= 19))
    ).astype(int)

    weather_map = {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3, "snow": 4}
    df["weather_severity_score"] = df["weather_condition"].map(weather_map).fillna(1).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    df["demand_weather_interaction"] = df["passenger_demand"] * df["weather_severity_score"]
    df["fare_per_driver"] = df["base_fare"] / (df["driver_availability"] + 1)

    # Encode weather
    df, encoder = encode_categoricals(df, ["weather_condition"], fit=True)

    X = df.drop(columns=["price_multiplier"])
    y = df["price_multiplier"]

    return X, y, encoder


@pytest.fixture
def sample_request():
    """Sample API request data."""
    return {
        "hour_of_day": 18,
        "day_of_week": 2,
        "is_weekend": False,
        "passenger_demand": 50,
        "driver_availability": 10,
        "weather_condition": "rain",
        "temperature": 12.5,
        "visibility_km": 5.0,
        "base_fare": 8.50,
    }


class TestPricePredictor:
    """Tests for the PricePredictor class."""

    def test_preprocess_adds_engineered_features(self, sample_request):
        """Preprocessing should add all engineered features."""
        predictor = PricePredictor.__new__(PricePredictor)
        predictor.scaler = None
        predictor.encoder = None
        predictor.feature_names = None
        predictor.config = {"data": {"target_column": "price_multiplier"}}

        result = predictor.preprocess(sample_request)

        assert "demand_supply_ratio" in result.columns
        assert "is_rush_hour" in result.columns
        assert "weather_severity_score" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "is_night" in result.columns
        assert "demand_weather_interaction" in result.columns
        assert "fare_per_driver" in result.columns

    def test_preprocess_drops_non_features(self, sample_request):
        """Preprocessing should drop non-feature columns (ride_id, timestamp)."""
        predictor = PricePredictor.__new__(PricePredictor)
        predictor.scaler = None
        predictor.encoder = None
        predictor.feature_names = None
        predictor.config = {"data": {"target_column": "price_multiplier"}}

        result = predictor.preprocess(sample_request)

        assert "ride_id" not in result.columns
        assert "timestamp" not in result.columns
        # weather_condition stays if no encoder is available (encoded separately during training)

    def test_preprocess_output_shape(self, sample_request):
        """Preprocessing should return single-row DataFrame."""
        predictor = PricePredictor.__new__(PricePredictor)
        predictor.scaler = None
        predictor.encoder = None
        predictor.feature_names = None
        predictor.config = {"data": {"target_column": "price_multiplier"}}

        result = predictor.preprocess(sample_request)
        assert result.shape[0] == 1

    def test_demand_supply_ratio_calculation(self):
        """Demand/supply ratio should be calculated correctly."""
        predictor = PricePredictor.__new__(PricePredictor)
        predictor.scaler = None
        predictor.encoder = None
        predictor.feature_names = None
        predictor.config = {"data": {"target_column": "price_multiplier"}}

        request = {
            "hour_of_day": 12,
            "day_of_week": 0,
            "is_weekend": False,
            "passenger_demand": 100,
            "driver_availability": 10,
            "weather_condition": "clear",
            "temperature": 20.0,
            "visibility_km": 10.0,
            "base_fare": 5.0,
        }

        result = predictor.preprocess(request)
        expected_ratio = 100 / 11  # demand / (drivers + 1)
        assert abs(result["demand_supply_ratio"].iloc[0] - expected_ratio) < 0.01

    def test_rush_hour_detection(self):
        """Rush hour should be detected correctly."""
        predictor = PricePredictor.__new__(PricePredictor)
        predictor.scaler = None
        predictor.encoder = None
        predictor.feature_names = None
        predictor.config = {"data": {"target_column": "price_multiplier"}}

        # Rush hour (6 PM)
        request_rush = {
            "hour_of_day": 18,
            "day_of_week": 0,
            "is_weekend": False,
            "passenger_demand": 50,
            "driver_availability": 10,
            "weather_condition": "clear",
            "temperature": 20.0,
            "visibility_km": 10.0,
            "base_fare": 5.0,
        }
        result = predictor.preprocess(request_rush)
        assert result["is_rush_hour"].iloc[0] == 1

        # Not rush hour (2 PM)
        request_non_rush = request_rush.copy()
        request_non_rush["hour_of_day"] = 14
        result = predictor.preprocess(request_non_rush)
        assert result["is_rush_hour"].iloc[0] == 0

    def test_night_detection(self):
        """Night time should be detected correctly."""
        predictor = PricePredictor.__new__(PricePredictor)
        predictor.scaler = None
        predictor.encoder = None
        predictor.feature_names = None
        predictor.config = {"data": {"target_column": "price_multiplier"}}

        request = {
            "hour_of_day": 2,
            "day_of_week": 0,
            "is_weekend": False,
            "passenger_demand": 10,
            "driver_availability": 5,
            "weather_condition": "clear",
            "temperature": 10.0,
            "visibility_km": 10.0,
            "base_fare": 5.0,
        }
        result = predictor.preprocess(request)
        assert result["is_night"].iloc[0] == 1

    def test_weather_severity_mapping(self):
        """Weather severity should be mapped correctly."""
        predictor = PricePredictor.__new__(PricePredictor)
        predictor.scaler = None
        predictor.encoder = None
        predictor.feature_names = None
        predictor.config = {"data": {"target_column": "price_multiplier"}}

        weather_scores = {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3, "snow": 4}

        for weather, expected_score in weather_scores.items():
            request = {
                "hour_of_day": 12,
                "day_of_week": 0,
                "is_weekend": False,
                "passenger_demand": 30,
                "driver_availability": 15,
                "weather_condition": weather,
                "temperature": 20.0,
                "visibility_km": 10.0,
                "base_fare": 5.0,
            }
            result = predictor.preprocess(request)
            assert result["weather_severity_score"].iloc[0] == expected_score


class TestEvaluateModelFunction:
    """Tests for the evaluation helper."""

    def test_perfect_predictions(self):
        """Perfect predictions should give R2=1, RMSE=0."""
        from src.models.train import evaluate_model

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = evaluate_model(y_true, y_pred)

        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)

    def test_known_rmse(self):
        """Should calculate RMSE correctly."""
        from src.models.train import evaluate_model

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        metrics = evaluate_model(y_true, y_pred)

        expected_rmse = np.sqrt(((0.5**2 + 0.5**2 + 0.5**2) / 3))
        assert metrics["rmse"] == pytest.approx(expected_rmse)

    def test_worse_than_mean(self):
        """Predictions worse than mean should give negative R2."""
        from src.models.train import evaluate_model

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Inverted
        metrics = evaluate_model(y_true, y_pred)

        assert metrics["r2"] < 0

    def test_metrics_are_floats(self):
        """Metrics should be Python floats."""
        from src.models.train import evaluate_model

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        metrics = evaluate_model(y_true, y_pred)

        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["r2"], float)
