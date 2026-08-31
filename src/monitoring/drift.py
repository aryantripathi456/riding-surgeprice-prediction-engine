"""Data drift detection using Evidently AI.

Compares incoming data against the training reference data to detect
distribution shifts that could degrade model performance.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.utils import load_config

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift between reference and current datasets."""

    def __init__(self, config: dict = None):
        """Initialize the drift detector."""
        if config is None:
            config = load_config()

        self.config = config
        self.reference_data = None
        self.current_data = None
        self.snapshot = None

        self._load_reference_data()

    def _load_reference_data(self):
        """Load the training data as reference dataset."""
        train_path = self.config["paths"]["train_features"]
        if Path(train_path).exists():
            self.reference_data = pd.read_csv(train_path)
            logger.info("Reference data loaded: %d rows, %d columns",
                       len(self.reference_data), len(self.reference_data.columns))
        else:
            logger.warning("Reference data not found at %s", train_path)

    def set_current_data(self, data: pd.DataFrame):
        """Set the current (incoming) data for comparison."""
        self.current_data = data
        logger.info("Current data set: %d rows", len(data))

    def generate_report(self, save_html: bool = True) -> dict:
        """Generate a full drift report comparing reference vs current data.

        Returns:
            Dictionary with drift metrics summary.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded")
        if self.current_data is None:
            raise ValueError("Current data not set. Call set_current_data() first.")

        # Create Evidently report
        report = Report(metrics=[DataDriftPreset()])
        self.snapshot = report.run(self.reference_data, self.current_data)

        # Extract results
        result_dict = self.snapshot.dump_dict()
        summary = self._extract_summary(result_dict)

        # Save outputs
        if save_html:
            self._save_report()
        self._save_summary(summary)

        return summary

    def _extract_summary(self, result_dict: dict) -> dict:
        """Extract key metrics from the report dump."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_rows": len(self.reference_data),
            "current_rows": len(self.current_data),
            "dataset_drift": False,
            "n_drifted_features": 0,
            "drifted_features": [],
            "drift_scores": {},
        }

        metric_results = result_dict.get("metric_results", {})

        for _key, metric in metric_results.items():
            display_name = metric.get("display_name", "")
            metric_type = metric.get("type", "")

            # Count of drifted columns
            if "Count of Drifted Columns" in display_name:
                count_data = metric.get("count", {})
                n_drifted = count_data.get("value", 0)
                summary["n_drifted_features"] = int(n_drifted)
                summary["dataset_drift"] = n_drifted > 0

            # Per-column drift
            elif "Value drift for" in display_name:
                column_name = display_name.replace("Value drift for ", "")

                # Extract drift info from widget counters
                widgets = metric.get("widget", [])
                drift_detected = False
                drift_score = 0.0

                for widget in widgets:
                    params = widget.get("params") or {}
                    counters = params.get("counters", [])
                    for counter in counters:
                        label = counter.get("label", "")
                        value = counter.get("value", "")

                        if "Data drift detected" in label:
                            drift_detected = True
                        elif "Data drift not detected" in label:
                            drift_detected = False

                        # Extract drift score from label
                        score_match = re.search(r"Drift score: ([0-9.]+)", label)
                        if score_match:
                            drift_score = float(score_match.group(1))

                summary["drift_scores"][column_name] = {
                    "drift_detected": drift_detected,
                    "drift_score": drift_score,
                }

                if drift_detected:
                    summary["drifted_features"].append(column_name)

        return summary

    def _save_report(self):
        """Save the Evidently report as HTML."""
        output_dir = Path(self.config["paths"]["drift_reports"])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"drift_report_{timestamp}.html"

        self.snapshot.save_html(filepath)
        logger.info("Drift report saved to %s", filepath)

    def _save_summary(self, summary: dict):
        """Save the summary as JSON."""
        output_dir = Path(self.config["paths"]["drift_reports"])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"drift_summary_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Drift summary saved to %s", filepath)


def main():
    """Run drift detection on reference vs synthetic current data."""
    print("=" * 60)
    print("DATA DRIFT DETECTION")
    print("=" * 60)

    detector = DriftDetector()

    if detector.reference_data is None:
        print("ERROR: Reference data not found. Run the data pipeline first.")
        return

    # Generate synthetic current data (simulating normal distribution)
    import numpy as np
    reference = detector.reference_data
    rng = np.random.default_rng(42)
    current = reference.sample(n=min(2000, len(reference)), random_state=42).copy()

    # Add small noise to simulate mild drift
    numeric_cols = current.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "price_multiplier":
            std = current[col].std()
            current[col] += rng.normal(0, std * 0.1, size=len(current))

    print(f"\nReference data: {len(reference)} rows")
    print(f"Current data:   {len(current)} rows")

    detector.set_current_data(current)

    # Generate report
    print("\nGenerating drift report...")
    summary = detector.generate_report(save_html=True)

    print(f"\n--- Drift Summary ---")
    print(f"Dataset drift detected: {summary['dataset_drift']}")
    print(f"Number of drifted features: {summary['n_drifted_features']}")
    print(f"Drifted features: {summary['drifted_features']}")

    if summary["drift_scores"]:
        print(f"\nDrift scores per feature:")
        for feature, scores in summary["drift_scores"].items():
            status = "DRIFT" if scores["drift_detected"] else "OK"
            print(f"  {feature}: {scores['drift_score']:.4f} [{status}]")


if __name__ == "__main__":
    main()
