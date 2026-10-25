"""Train the six-input model used by the Streamlit prediction form.

Run from the project root with:
    python train_prediction_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_diabetic_data.csv"
MODEL_PATH = ROOT / "data" / "processed" / "prediction_model_6_features.joblib"

FEATURES = [
    "time_in_hospital",
    "num_medications",
    "number_inpatient",
    "number_outpatient",
    "number_emergency",
    "num_lab_procedures",
]
TARGET = "readmitted_30"


def main():
    data = pd.read_csv(DATA_PATH)

    missing = [column for column in FEATURES + [TARGET] if column not in data.columns]
    if missing:
        raise ValueError(
            f"Required columns are missing from {DATA_PATH.name}: {missing}"
        )

    data = data[FEATURES + [TARGET]].dropna()
    X = data[FEATURES].astype(float)
    y = data[TARGET].astype(int)

    # Keep the final test data separate from training and threshold selection.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.05,
        max_iter=150,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Select a decision threshold using validation data only.
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    thresholds = np.arange(0.10, 0.901, 0.01)
    threshold_scores = [
        f1_score(
            y_validation,
            validation_probabilities >= threshold,
            zero_division=0,
        )
        for threshold in thresholds
    ]
    threshold = float(thresholds[int(np.argmax(threshold_scores))])

    # Report final performance on the held-out test data.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "test_precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "test_f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "test_readmission_rate": float(y_test.mean()),
    }

    artifact = {
        "model": model,
        "feature_columns": FEATURES,
        "optimal_threshold": threshold,
        "threshold_selection": "maximum F1 on validation partition",
        "metrics": metrics,
        "disclaimer": (
            "Educational analysis only; not validated for clinical decisions."
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved compatible six-input model: {MODEL_PATH}")
    print(f"Decision threshold selected on validation data: {threshold:.2f}")
    print("Held-out test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()