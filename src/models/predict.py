"""Prediction module for the dynamic pricing engine.

Loads the trained model and provides inference functionality.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
# from sklearn.base import BaseEstimator

from src.utils import load_config


class PricePredictor:
    """Wrapper for making price multiplier predictions."""

    def __init__(self, model_path: str = None, config: dict = None):
        """Load model and preprocessing artifacts."""
        if config is None:
            config = load_config()

        if model_path is None:
            model_path = config["paths"]["model_output"]

        self.model = joblib.load(model_path)
        self.config = config

        # Store expected feature names from the model
        self.feature_names = list(self.model.get_booster().feature_names) if hasattr(self.model, 'get_booster') else None

        # Load scaler and encoder if available
        scaler_path = Path("models/scaler.pkl")
        encoder_path = Path("models/encoder.pkl")

        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.encoder = joblib.load(encoder_path) if encoder_path.exists() else None

    def preprocess(self, data: dict) -> pd.DataFrame:
        """Preprocess input data for prediction.

        Args:
            data: Dictionary with ride request features.

        Returns:
            Preprocessed DataFrame ready for model input.
        """
        df = pd.DataFrame([data])

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

        # Encode weather condition
        if self.encoder is not None and "weather_condition" in df.columns:
            weather_encoded = self.encoder.transform(df[["weather_condition"]])
            weather_cols = self.encoder.get_feature_names_out(["weather_condition"])
            weather_df = pd.DataFrame(weather_encoded, columns=weather_cols, index=df.index)
            df = pd.concat([df.drop(columns=["weather_condition"]), weather_df], axis=1)

        # Drop non-feature columns
        cols_to_drop = ["ride_id", "timestamp"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        # Ensure columns are in the same order as training
        if self.feature_names:
            df = df[self.feature_names]

        return df

    def predict(self, data: dict) -> dict:
        """Make a price multiplier prediction.

        Args:
            data: Dictionary with ride request features.

        Returns:
            Dictionary with prediction results.
        """
        processed = self.preprocess(data)

        prediction = self.model.predict(processed)[0]
        prediction = float(np.clip(prediction, 1.0, 3.0))

        base_fare = data.get("base_fare", 10.0)
        estimated_price = base_fare * prediction

        # Confidence based on prediction range
        if 1.3 <= prediction <= 2.0:
            confidence = "medium"
        elif prediction < 1.3 or prediction > 2.5:
            confidence = "low"
        else:
            confidence = "high"

        return {
            "price_multiplier": round(prediction, 2),
            "estimated_price": round(estimated_price, 2),
            "confidence": confidence,
        }


def main():
    """Test prediction with sample data."""
    sample = {
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

    predictor = PricePredictor()
    result = predictor.predict(sample)

    print("Sample prediction:")
    print(f"  Input: {sample}")
    print(f"  Result: {result}")


if __name__ == "__main__":
    main()
