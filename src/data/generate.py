"""Synthetic ride data generation for the dynamic pricing engine.

Generates 100K ride request records with realistic patterns for
demand, supply, weather, and pricing.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_config


def generate_synthetic_data(n_samples: int = 100_000, random_seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ride request data.

    Args:
        n_samples: Number of records to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with ride request data including the target price_multiplier.
    """
    rng = np.random.default_rng(random_seed)

    # --- Time features ---
    ride_ids = np.arange(1, n_samples + 1)

    # Generate timestamps over 90 days
    start_date = pd.Timestamp("2024-01-01")
    random_offsets = rng.integers(0, 90 * 24 * 3600, size=n_samples)
    timestamps = pd.to_datetime(start_date) + pd.to_timedelta(random_offsets, unit="s")

    hour_of_day = timestamps.hour
    day_of_week = timestamps.dayofweek
    is_weekend = day_of_week >= 5

    # --- Demand & Supply ---
    # Demand pattern: peaks during rush hours, higher on weekends
    base_demand = rng.poisson(lam=30, size=n_samples).astype(float)
    rush_hour_mask = ((hour_of_day >= 7) & (hour_of_day <= 9)) | (
        (hour_of_day >= 17) & (hour_of_day <= 19)
    )
    base_demand[rush_hour_mask] *= rng.uniform(1.5, 2.5, size=rush_hour_mask.sum())
    base_demand[is_weekend] *= rng.uniform(1.1, 1.8, size=is_weekend.sum())
    # Late night (0-5 AM): lower demand
    late_night_mask = (hour_of_day >= 0) & (hour_of_day <= 5)
    base_demand[late_night_mask] *= rng.uniform(0.3, 0.6, size=late_night_mask.sum())
    passenger_demand = np.clip(base_demand, 1, 200).astype(int)

    # Driver availability: inversely related to time-of-day demand
    base_drivers = rng.poisson(lam=15, size=n_samples).astype(float)
    base_drivers[rush_hour_mask] *= rng.uniform(0.4, 0.7, size=rush_hour_mask.sum())
    base_drivers[late_night_mask] *= rng.uniform(0.5, 0.8, size=late_night_mask.sum())
    base_drivers[is_weekend] *= rng.uniform(0.7, 1.0, size=is_weekend.sum())
    driver_availability = np.clip(base_drivers, 1, 100).astype(int)

    # --- Weather ---
    weather_choices = ["clear", "cloudy", "rain", "storm", "snow"]
    # Seasonal weather probabilities (approximate)
    month = timestamps.month
    weather_probs = np.zeros((n_samples, len(weather_choices)))

    for i in range(n_samples):
        m = month[i]
        if m in [6, 7, 8]:  # Summer
            weather_probs[i] = [0.5, 0.25, 0.15, 0.08, 0.02]
        elif m in [12, 1, 2]:  # Winter
            weather_probs[i] = [0.2, 0.25, 0.2, 0.15, 0.2]
        elif m in [3, 4, 5]:  # Spring
            weather_probs[i] = [0.35, 0.3, 0.2, 0.1, 0.05]
        else:  # Fall
            weather_probs[i] = [0.3, 0.3, 0.25, 0.1, 0.05]

    weather_idx = np.array(
        [rng.choice(len(weather_choices), p=weather_probs[i]) for i in range(n_samples)]
    )
    weather_condition = [weather_choices[idx] for idx in weather_idx]

    # Temperature correlated with month and weather
    base_temp = 15 + 15 * np.sin(2 * np.pi * (month - 3) / 12)  # Seasonal cycle
    weather_temp_offset = {0: 3, 1: 0, 2: -3, 3: -7, 4: -10}
    temperature = base_temp + np.array([weather_temp_offset[w] for w in weather_idx])
    temperature += rng.normal(0, 3, size=n_samples)  # Noise
    temperature = np.round(temperature, 1)

    # Visibility correlated with weather
    base_visibility = np.where(
        weather_idx == 0, 15.0,  # clear
        np.where(weather_idx == 1, 12.0,  # cloudy
        np.where(weather_idx == 2, 6.0,  # rain
        np.where(weather_idx == 3, 2.5,   # storm
        4.0))),  # snow
    )
    visibility_km = base_visibility + rng.normal(0, 1.5, size=n_samples)
    visibility_km = np.round(np.clip(visibility_km, 0.1, 20.0), 1)

    # --- Base fare ---
    # Varies by time of day (surge-free baseline)
    base_fare = 5.0 + 2.0 * np.sin(2 * np.pi * hour_of_day / 24)
    base_fare += rng.uniform(-0.5, 0.5, size=n_samples)
    base_fare = np.round(np.clip(base_fare, 3.0, 12.0), 2)

    # --- Target: price_multiplier ---
    # Derived from demand/supply ratio, weather severity, and time patterns
    demand_supply_ratio = passenger_demand / (driver_availability + 1)

    # Weather severity: 0 (clear) to 4 (storm)
    weather_severity_map = {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3, "snow": 4}
    weather_severity = np.array([weather_severity_map[w] for w in weather_condition])

    # Price multiplier formula with noise
    multiplier = (
        1.0
        + 0.3 * (demand_supply_ratio - 1.0)  # Demand/supply effect
        + 0.08 * weather_severity              # Weather effect
        + 0.15 * rush_hour_mask.astype(float)  # Rush hour premium
        + 0.05 * is_weekend.astype(float)      # Weekend slight premium
        + rng.normal(0, 0.05, size=n_samples)  # Random noise
    )
    price_multiplier = np.round(np.clip(multiplier, 1.0, 3.0), 2)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "ride_id": ride_ids,
        "timestamp": timestamps,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "passenger_demand": passenger_demand,
        "driver_availability": driver_availability,
        "weather_condition": weather_condition,
        "temperature": temperature,
        "visibility_km": visibility_km,
        "base_fare": base_fare,
        "price_multiplier": price_multiplier,
    })

    return df


def main():
    """Generate synthetic data and save to CSV."""
    config = load_config()
    n_samples = config["data"]["n_samples"]
    random_seed = config["data"]["random_seed"]

    print(f"Generating {n_samples:,} synthetic ride records...")
    df = generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)

    output_path = Path(config["paths"]["raw_data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nTarget (price_multiplier) stats:")
    print(df["price_multiplier"].describe().round(3))
    print(f"\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
