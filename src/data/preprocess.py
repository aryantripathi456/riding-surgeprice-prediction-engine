"""Data preprocessing pipeline for the dynamic pricing engine.

Handles missing values, encoding, scaling, and train/test split.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import load_config


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw data from CSV."""
    return pd.read_csv(filepath, parse_dates=["timestamp"])


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset.

    - Numeric columns: fill with median
    - Categorical columns: fill with mode
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not features for modeling."""
    cols_to_drop = ["ride_id", "timestamp"]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing_cols)


def encode_categoricals(
    df: pd.DataFrame, categorical_columns: list, encoder: OneHotEncoder = None, fit: bool = True
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode categorical columns.

    Returns the transformed DataFrame and fitted encoder.
    """
    df = df.copy()

    if fit:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_data = encoder.fit_transform(df[categorical_columns])
    else:
        encoded_data = encoder.transform(df[categorical_columns])

    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(categorical_columns), index=df.index
    )

    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder


def scale_features(
    df: pd.DataFrame, feature_columns: list, scaler: StandardScaler = None, fit: bool = True
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler.

    Returns the transformed DataFrame and fitted scaler.
    """
    df = df.copy()

    if fit:
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
    else:
        df[feature_columns] = scaler.transform(df[feature_columns])

    return df, scaler


def preprocess_data(config: dict = None) -> dict:
    """Run the full preprocessing pipeline.

    Steps:
        1. Load raw data
        2. Handle missing values
        3. Drop unnecessary columns
        4. Encode categoricals
        5. Scale features
        6. Train/test split

    Returns:
        Dictionary with train/test splits and artifacts.
    """
    if config is None:
        config = load_config()

    target_col = config["data"]["target_column"]
    test_size = config["data"]["test_size"]
    random_seed = config["data"]["random_seed"]
    categorical_columns = config["features"]["categorical_columns"]

    # Load
    print("Loading raw data...")
    df = load_raw_data(config["paths"]["raw_data"])
    print(f"  Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Clean
    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Dropping unnecessary columns...")
    df = drop_unnecessary_columns(df)

    # Encode
    print("Encoding categorical variables...")
    df, encoder = encode_categoricals(df, categorical_columns, fit=True)

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_columns = X.columns.tolist()

    # Train/test split
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    print(f"  Train: {X_train.shape[0]:,} rows")
    print(f"  Test:  {X_test.shape[0]:,} rows")

    # Scale
    print("Scaling features...")
    X_train, scaler = scale_features(X_train, feature_columns, fit=True)
    X_test, _ = scale_features(X_test, feature_columns, scaler=scaler, fit=False)

    # Reattach target
    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    test_df = X_test.copy()
    test_df[target_col] = y_test.values

    # Save processed data
    processed_path = Path(config["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    full_processed = pd.concat([train_df, test_df], axis=0)
    full_processed.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

    # Save train/test splits
    train_path = Path(config["paths"]["train_features"])
    test_path = Path(config["paths"]["test_features"])
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train saved to {train_path}")
    print(f"Test saved to {test_path}")

    # Save artifacts
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(encoder, models_dir / "encoder.pkl")
    print("Scaler and encoder saved to models/")

    return {
        "train": train_df,
        "test": test_df,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "encoder": encoder,
    }


def main():
    """Run preprocessing pipeline."""
    result = preprocess_data()

    print("\n--- Preprocessing Complete ---")
    print(f"Features: {len(result['feature_columns'])}")
    print(f"Train shape: {result['train'].shape}")
    print(f"Test shape: {result['test'].shape}")


if __name__ == "__main__":
    main()
