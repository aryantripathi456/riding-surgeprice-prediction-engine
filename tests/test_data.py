"""Comprehensive tests for data generation and preprocessing."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.data.generate import generate_synthetic_data
from src.data.preprocess import (
    handle_missing_values,
    drop_unnecessary_columns,
    encode_categoricals,
    scale_features,
)


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = generate_synthetic_data(n_samples=100, random_seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        """Should generate the requested number of rows."""
        for n in [10, 100, 1000]:
            df = generate_synthetic_data(n_samples=n)
            assert len(df) == n

    def test_correct_columns(self):
        """Should have all expected columns."""
        df = generate_synthetic_data(n_samples=100)
        expected_cols = [
            "ride_id", "timestamp", "hour_of_day", "day_of_week",
            "is_weekend", "passenger_demand", "driver_availability",
            "weather_condition", "temperature", "visibility_km",
            "base_fare", "price_multiplier",
        ]
        assert list(df.columns) == expected_cols

    def test_price_multiplier_range(self):
        """Price multiplier should be between 1.0 and 3.0."""
        df = generate_synthetic_data(n_samples=10000)
        assert df["price_multiplier"].min() >= 1.0
        assert df["price_multiplier"].max() <= 3.0

    def test_hour_of_day_range(self):
        """Hour of day should be 0-23."""
        df = generate_synthetic_data(n_samples=1000)
        assert df["hour_of_day"].min() >= 0
        assert df["hour_of_day"].max() <= 23

    def test_day_of_week_range(self):
        """Day of week should be 0-6."""
        df = generate_synthetic_data(n_samples=1000)
        assert df["day_of_week"].min() >= 0
        assert df["day_of_week"].max() <= 6

    def test_weather_conditions_valid(self):
        """Weather conditions should be from valid set."""
        valid_weather = {"clear", "cloudy", "rain", "storm", "snow"}
        df = generate_synthetic_data(n_samples=1000)
        assert set(df["weather_condition"].unique()).issubset(valid_weather)

    def test_reproducibility(self):
        """Same seed should produce same data."""
        df1 = generate_synthetic_data(n_samples=100, random_seed=42)
        df2 = generate_synthetic_data(n_samples=100, random_seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_data(self):
        """Different seeds should produce different data."""
        df1 = generate_synthetic_data(n_samples=100, random_seed=42)
        df2 = generate_synthetic_data(n_samples=100, random_seed=99)
        assert not df1.equals(df2)

    def test_no_missing_values(self):
        """Generated data should have no missing values."""
        df = generate_synthetic_data(n_samples=1000)
        assert df.isnull().sum().sum() == 0

    def test_temperature_range_reasonable(self):
        """Temperature should be within reasonable bounds."""
        df = generate_synthetic_data(n_samples=10000)
        assert df["temperature"].min() >= -30
        assert df["temperature"].max() <= 50

    def test_visibility_positive(self):
        """Visibility should be positive."""
        df = generate_synthetic_data(n_samples=1000)
        assert (df["visibility_km"] > 0).all()

    def test_base_fare_positive(self):
        """Base fare should be positive."""
        df = generate_synthetic_data(n_samples=1000)
        assert (df["base_fare"] > 0).all()

    def test_demand_positive(self):
        """Passenger demand should be positive."""
        df = generate_synthetic_data(n_samples=1000)
        assert (df["passenger_demand"] > 0).all()

    def test_drivers_positive(self):
        """Driver availability should be positive."""
        df = generate_synthetic_data(n_samples=1000)
        assert (df["driver_availability"] > 0).all()


class TestHandleMissingValues:
    """Tests for missing value handling."""

    def test_no_missing_values(self):
        """Should not modify data without missing values."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = handle_missing_values(df)
        pd.testing.assert_frame_equal(df, result)

    def test_fills_numeric_with_median(self):
        """Should fill numeric NaN with median."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})
        result = handle_missing_values(df)
        assert result["a"].isnull().sum() == 0
        assert result["a"].iloc[1] == 3.5  # median of [1, 3, 4, 5]

    def test_fills_categorical_with_mode(self):
        """Should fill categorical NaN with mode."""
        df = pd.DataFrame({"a": ["x", "y", "x", None, "y"]})
        result = handle_missing_values(df)
        assert result["a"].isnull().sum() == 0

    def test_preserves_non_missing_values(self):
        """Should not change non-missing values."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = handle_missing_values(df)
        pd.testing.assert_frame_equal(df, result)


class TestDropUnnecessaryColumns:
    """Tests for column dropping."""

    def test_drops_ride_id(self):
        """Should drop ride_id column."""
        df = pd.DataFrame({"ride_id": [1, 2], "value": [10, 20]})
        result = drop_unnecessary_columns(df)
        assert "ride_id" not in result.columns

    def test_drops_timestamp(self):
        """Should drop timestamp column."""
        df = pd.DataFrame({"timestamp": ["2024-01-01"], "value": [10]})
        result = drop_unnecessary_columns(df)
        assert "timestamp" not in result.columns

    def test_keeps_other_columns(self):
        """Should keep non-dropped columns."""
        df = pd.DataFrame({"ride_id": [1], "value": [10], "other": [5]})
        result = drop_unnecessary_columns(df)
        assert "value" in result.columns
        assert "other" in result.columns


class TestEncodeCategoricals:
    """Tests for categorical encoding."""

    def test_encodes_correctly(self):
        """Should one-hot encode categorical columns."""
        df = pd.DataFrame({"weather": ["clear", "rain", "clear", "snow"]})
        result, encoder = encode_categoricals(df, ["weather"], fit=True)
        assert "weather_clear" in result.columns
        assert "weather_rain" in result.columns
        assert "weather_snow" in result.columns
        assert "weather" not in result.columns

    def test_transform_uses_fitted_encoder(self):
        """Transform should use the fitted encoder."""
        df_train = pd.DataFrame({"weather": ["clear", "rain", "clear"]})
        df_test = pd.DataFrame({"weather": ["clear", "snow"]})

        _, encoder = encode_categoricals(df_train, ["weather"], fit=True)
        result_test, _ = encode_categoricals(df_test, ["weather"], encoder=encoder, fit=False)

        # "snow" is unknown to the encoder, so it gets all zeros with handle_unknown="ignore"
        assert "weather_clear" in result_test.columns
        assert result_test["weather_clear"].iloc[1] == 0.0  # snow maps to no column

    def test_preserves_numeric_columns(self):
        """Should not modify numeric columns."""
        df = pd.DataFrame({"weather": ["clear", "rain"], "value": [1.0, 2.0]})
        result, _ = encode_categoricals(df, ["weather"], fit=True)
        assert "value" in result.columns
        pd.testing.assert_series_equal(result["value"], df["value"], check_names=False)


class TestScaleFeatures:
    """Tests for feature scaling."""

    def test_scales_to_standard(self):
        """Should scale features to mean=0, std~1."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result, scaler = scale_features(df, ["a"], fit=True)
        assert abs(result["a"].mean()) < 1e-10
        # StandardScaler uses population std (ddof=0) = 1.0
        assert abs(result["a"].std(ddof=0) - 1.0) < 0.01

    def test_transform_uses_fitted_scaler(self):
        """Transform should use the fitted scaler."""
        df_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df_test = pd.DataFrame({"a": [10.0, 20.0]})

        _, scaler = scale_features(df_train, ["a"], fit=True)
        result_test, _ = scale_features(df_test, ["a"], scaler=scaler, fit=False)

        # Scaled test values should be far from 0 (since they're different distribution)
        assert abs(result_test["a"].mean()) > 1.0

    def test_preserves_shape(self):
        """Should preserve DataFrame shape."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result, _ = scale_features(df, ["a", "b"], fit=True)
        assert result.shape == df.shape
