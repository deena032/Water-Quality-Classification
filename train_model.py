"""
Water Quality Classification - Model Training Script
====================================================
Loads the real Water Potability dataset, handles missing values using median
imputation inside a scikit-learn Pipeline, splits data into training and test
sets, trains a Random Forest Classifier, evaluates performance, and saves the
trained pipeline to `model.pkl` using Joblib.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "water_potability.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET_COLUMN = "Potability"


def train_and_save_model() -> None:
    """Load dataset, train Random Forest pipeline, evaluate, and save artifacts."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please ensure 'water_potability.csv' is in the project root."
        )

    print("Loading Water Potability dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nMissing values per column before imputation:")
    print(df.isnull().sum().to_string())

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Stratified split preserves the potable / not-potable class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Pipeline ensures missing-value imputation is fitted ONLY on training data
    # and automatically applied at prediction time.
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining Random Forest Classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Potable (0)", "Potable (1)"],
    )

    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Save trained pipeline with Joblib
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model pipeline to: {MODEL_PATH}")

    # Save summary metrics for display in the web interface
    metrics = {
        "dataset_rows": int(df.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_accuracy": round(float(accuracy) * 100, 2),
        "roc_auc": round(float(roc_auc), 4),
        "potable_ratio": round(float(y.mean()) * 100, 2),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    train_and_save_model()
