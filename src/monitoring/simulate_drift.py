"""Drift simulation script for demonstrating monitoring capabilities.

Simulates a demand spike scenario and shows how the drift detection
system identifies distribution shifts in real-time.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.monitoring.drift import DriftDetector
from src.utils import load_config


def create_demand_spike_scenario(reference_data: pd.DataFrame, severity: str = "moderate") -> pd.DataFrame:
    """Create a synthetic scenario with a demand spike.

    Args:
        reference_data: The original training data to base the scenario on.
        severity: "mild", "moderate", or "severe".

    Returns:
        DataFrame simulating the demand spike scenario.
    """
    rng = np.random.default_rng(42)

    # Start with a sample from the reference
    scenario = reference_data.sample(n=min(5000, len(reference_data)), random_state=42).copy()

    # Define spike parameters
    severity_params = {
        "mild": {"demand_mult": 1.5, "driver_reduction": 0.8, "temp_drop": 2},
        "moderate": {"demand_mult": 2.5, "driver_reduction": 0.5, "temp_drop": 5},
        "severe": {"demand_mult": 4.0, "driver_reduction": 0.3, "temp_drop": 10},
    }
    params = severity_params[severity]

    # Spike demand
    if "passenger_demand" in scenario.columns:
        scenario["passenger_demand"] = (
            scenario["passenger_demand"] * params["demand_mult"]
            + rng.normal(0, 5, size=len(scenario))
        ).clip(lower=1).astype(int)

    # Reduce driver availability
    if "driver_availability" in scenario.columns:
        scenario["driver_availability"] = (
            scenario["driver_availability"] * params["driver_reduction"]
            + rng.normal(0, 2, size=len(scenario))
        ).clip(lower=1).astype(int)

    # Temperature drop (simulating sudden weather change)
    if "temperature" in scenario.columns:
        scenario["temperature"] -= params["temp_drop"]

    # Recalculate derived features
    if "passenger_demand" in scenario.columns and "driver_availability" in scenario.columns:
        scenario["demand_supply_ratio"] = scenario["passenger_demand"] / (scenario["driver_availability"] + 1)

    return scenario


def create_weather_shift_scenario(reference_data: pd.DataFrame) -> pd.DataFrame:
    """Create a scenario where weather shifts to severe conditions."""
    scenario = reference_data.sample(n=min(3000, len(reference_data)), random_state=99).copy()

    # Shift weather to storm/snow
    if "weather_condition_storm" in scenario.columns:
        scenario["weather_condition_storm"] = 1
        scenario["weather_condition_clear"] = 0
        scenario["weather_condition_rain"] = 0
        scenario["weather_condition_cloudy"] = 0
        if "weather_condition_snow" in scenario.columns:
            scenario["weather_condition_snow"] = 0

    if "weather_severity_score" in scenario.columns:
        scenario["weather_severity_score"] = 3  # storm level

    return scenario


def run_simulation():
    """Run the full drift simulation."""
    config = load_config()
    output_dir = Path(config["paths"]["drift_reports"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DRIFT SIMULATION - Production Monitoring Demo")
    print("=" * 70)

    # Load reference data
    detector = DriftDetector(config=config)
    if detector.reference_data is None:
        print("ERROR: Reference data not found. Run the data pipeline first.")
        return

    reference = detector.reference_data
    print(f"\nReference dataset: {len(reference):,} rows")
    print(f"Features: {len(reference.columns)}")

    scenarios = [
        ("No Drift (Control)", reference.sample(n=2000, random_state=42)),
        ("Mild Demand Spike", create_demand_spike_scenario(reference, "mild")),
        ("Moderate Demand Spike", create_demand_spike_scenario(reference, "moderate")),
        ("Severe Demand Spike", create_demand_spike_scenario(reference, "severe")),
        ("Weather Shift to Storms", create_weather_shift_scenario(reference)),
    ]

    results = []

    for name, current_data in scenarios:
        print(f"\n{'─' * 70}")
        print(f"SCENARIO: {name}")
        print(f"{'─' * 70}")
        print(f"Current data: {len(current_data):,} rows")

        detector.set_current_data(current_data)

        # Generate report
        try:
            summary = detector.generate_report(save_html=True)

            result = {
                "scenario": name,
                "dataset_drift": summary["dataset_drift"],
                "n_drifted_features": summary["n_drifted_features"],
                "drifted_features": summary["drifted_features"],
                "drift_scores": summary["drift_scores"],
            }
            results.append(result)

            # Print summary
            drift_status = "YES" if summary["dataset_drift"] else "NO"
            print(f"  Drift detected: {drift_status}")
            print(f"  Drifted features: {summary['n_drifted_features']}")
            if summary["drifted_features"]:
                print(f"  Features with drift: {', '.join(summary['drifted_features'][:5])}")
                if len(summary["drifted_features"]) > 5:
                    print(f"    ... and {len(summary['drifted_features']) - 5} more")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"scenario": name, "error": str(e)})

    # Save combined results
    results_path = output_dir / "simulation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSimulation results saved to {results_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SIMULATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Scenario':<30} {'Drift?':<10} {'Features':<10}")
    print(f"{'─' * 50}")
    for r in results:
        if "error" in r:
            print(f"{r['scenario']:<30} {'ERROR':<10} {r['error'][:30]}")
        else:
            drift = "YES" if r["dataset_drift"] else "NO"
            print(f"{r['scenario']:<30} {drift:<10} {r['n_drifted_features']:<10}")

    print(f"\nHTML reports saved to: {output_dir}")
    print("Open any .html file in a browser to view the interactive drift report.")


if __name__ == "__main__":
    run_simulation()
