"""Feature engineering pipeline for the dynamic pricing engine.

Creates derived features from the preprocessed data to improve model performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_config


def add_demand_supply_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand-to-supply ratio feature."""
    df = df.copy()
    df["demand_supply_ratio"] = df["passenger_demand"] / (df["driver_availability"] + 1)
    return df


def add_rush_hour_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flag for rush hours (7-9 AM, 5-7 PM)."""
    df = df.copy()
    df["is_rush_hour"] = (
        ((df["hour_of_day"] >= 7) & (df["hour_of_day"] <= 9))
        | ((df["hour_of_day"] >= 17) & (df["hour_of_day"] <= 19))
    ).astype(int)
    return df


def add_weather_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Add ordinal weather severity score."""
    df = df.copy()
    # If weather columns are already one-hot encoded, we may not have the raw column.
    # The weather severity will be added during feature building before encoding,
    # or reconstructed from one-hot columns if needed.
    weather_map = {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3, "snow": 4}

    if "weather_condition" in df.columns:
        df["weather_severity_score"] = df["weather_condition"].map(weather_map).fillna(1).astype(int)
    else:
        # Reconstruct from one-hot encoded columns if available
        for weather, score in weather_map.items():
            col_name = f"weather_condition_{weather}"
            if col_name in df.columns:
                if "weather_severity_score" not in df.columns:
                    df["weather_severity_score"] = (df[col_name] * score).astype(int)
                else:
                    df["weather_severity_score"] += (df[col_name] * score).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add additional time-based features."""
    df = df.copy()
    # Part of day encoding (cyclical)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    # Is nighttime (10 PM - 5 AM)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between key variables."""
    df = df.copy()
    # Demand * weather severity interaction
    if "weather_severity_score" in df.columns:
        df["demand_weather_interaction"] = df["passenger_demand"] * df["weather_severity_score"]

    # Base fare per driver (supply-adjusted pricing)
    df["fare_per_driver"] = df["base_fare"] / (df["driver_availability"] + 1)

    return df


def build_features(config: dict = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the feature engineering pipeline on train and test data.

    Returns:
        Tuple of (train_df, test_df) with engineered features.
    """
    if config is None:
        config = load_config()

    # Load train/test splits
    print("Loading train/test data...")
    train_df = pd.read_csv(config["paths"]["train_features"])
    test_df = pd.read_csv(config["paths"]["test_features"])

    print(f"  Train: {train_df.shape}")
    print(f"  Test:  {test_df.shape}")

    # Apply feature engineering to both splits
    for df_name, df in [("train", train_df), ("test", test_df)]:
        print(f"Building features for {df_name}...")

    # Build features for both
    train_df = add_demand_supply_ratio(train_df)
    train_df = add_rush_hour_flag(train_df)
    train_df = add_weather_severity(train_df)
    train_df = add_time_features(train_df)
    train_df = add_interaction_features(train_df)

    test_df = add_demand_supply_ratio(test_df)
    test_df = add_rush_hour_flag(test_df)
    test_df = add_weather_severity(test_df)
    test_df = add_time_features(test_df)
    test_df = add_interaction_features(test_df)

    # Save featured data
    train_path = Path(config["paths"]["train_features"])
    test_path = Path(config["paths"]["test_features"])

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nFeatured train saved to {train_path}: {train_df.shape}")
    print(f"Featured test saved to {test_path}: {test_df.shape}")

    return train_df, test_df


def main():
    """Run feature engineering pipeline."""
    train_df, test_df = build_features()

    print("\n--- Feature Engineering Complete ---")
    print(f"Train columns ({len(train_df.columns)}): {list(train_df.columns)}")


if __name__ == "__main__":
    main()
