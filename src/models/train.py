"""Model training pipeline with MLflow experiment tracking.

Trains multiple regression models, compares performance, and registers the best model.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.utils import load_config

warnings.filterwarnings("ignore", category=UserWarning)

# Model class mapping
MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
    "LGBMRegressor": LGBMRegressor,
}


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets."""
    train_df = pd.read_csv(config["paths"]["train_features"])
    test_df = pd.read_csv(config["paths"]["test_features"])

    target_col = config["data"]["target_column"]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def train_single_model(
    model_name: str,
    model_class_name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str,
) -> dict:
    """Train a single model and log to MLflow.

    Returns:
        Dictionary with model name, metrics, training time, and fitted model.
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"  Class: {model_class_name}")
    print(f"  Params: {params}")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        # Initialize model
        model_class = MODEL_CLASSES[model_class_name]
        model = model_class(**params)

        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)

        print(f"  Training time: {training_time:.2f}s")
        print(f"  Train RMSE: {train_metrics['rmse']:.4f} | R2: {train_metrics['r2']:.4f}")
        print(f"  Test  RMSE: {test_metrics['rmse']:.4f} | R2: {test_metrics['r2']:.4f}")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_class", model_class_name)
        mlflow.log_metric("training_time", training_time)

        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Log model artifact (handle different model types)
        if model_class_name in ("XGBRegressor",):
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif model_class_name in ("LGBMRegressor",):
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    return {
        "model_name": model_name,
        "model_class": model_class_name,
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "training_time": training_time,
        "run_id": run_id,
    }


def train_all_models(config: dict) -> list[dict]:
    """Train all configured models and return results.

    Returns:
        List of result dictionaries sorted by test RMSE.
    """
    X_train, X_test, y_train, y_test = load_data(config)
    experiment_name = config["mlflow"]["experiment_name"]

    print(f"Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Test data:     {X_test.shape[0]:,} samples")
    print(f"MLflow experiment: {experiment_name}")

    results = []

    for model_config in config["model"]["experiments"]:
        name = model_config["name"]
        class_name = model_config["class"]
        params = model_config["params"]

        result = train_single_model(
            model_name=name,
            model_class_name=class_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            experiment_name=experiment_name,
        )
        results.append(result)

    # Sort by test RMSE (lower is better)
    results.sort(key=lambda x: x["test_metrics"]["rmse"])

    return results


def save_comparison_table(results: list[dict], config: dict) -> pd.DataFrame:
    """Save a comparison table of all models."""
    rows = []
    for r in results:
        row = {
            "model_name": r["model_name"],
            "model_class": r["model_class"],
            "train_rmse": r["train_metrics"]["rmse"],
            "train_r2": r["train_metrics"]["r2"],
            "test_rmse": r["test_metrics"]["rmse"],
            "test_r2": r["test_metrics"]["r2"],
            "test_mae": r["test_metrics"]["mae"],
            "training_time_s": r["training_time"],
            "run_id": r["run_id"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(config["paths"]["model_comparison"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nModel comparison saved to {output_path}")

    return df


def save_best_model(results: list[dict], config: dict) -> None:
    """Save the best model (lowest test RMSE) to disk."""
    best = results[0]
    model_path = Path(config["paths"]["model_output"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best["model"], model_path)
    print(f"Best model ({best['model_name']}) saved to {model_path}")
    print(f"  Test RMSE: {best['test_metrics']['rmse']:.4f}")
    print(f"  Test R2:   {best['test_metrics']['r2']:.4f}")

    # Save metrics
    metrics = {
        "best_model": best["model_name"],
        "test_rmse": best["test_metrics"]["rmse"],
        "test_r2": best["test_metrics"]["r2"],
        "test_mae": best["test_metrics"]["mae"],
        "training_time": best["training_time"],
    }

    metrics_path = Path(config["paths"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    """Run the complete model training pipeline."""
    config = load_config()

    print("=" * 60)
    print("DYNAMIC PRICING ENGINE - MODEL TRAINING")
    print("=" * 60)

    results = train_all_models(config)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by test RMSE)")
    print("=" * 60)

    comparison_df = save_comparison_table(results, config)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    save_best_model(results, config)

    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
