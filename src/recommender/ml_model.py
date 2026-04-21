"""ML-based nutrition recommender using logistic regression."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "nutrition_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "nutrition_logreg.pkl"

ACTIVITY_MAP = {"low": 0, "medium": 1, "moderate": 1, "high": 2, "very active": 2}
SUGAR_MAP = {"low": 0, "high": 1}

FEATURE_COLUMNS = ["age", "weight", "activity_level_encoded", "sugar_preference_encoded"]
TARGET_COLUMN = "diet_label"


@dataclass
class ModelBundle:
    model: LogisticRegression
    feature_columns: list[str]
    class_labels: list[str]


def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    processed["activity_level_encoded"] = (
        processed["activity_level"].astype(str).str.lower().map(ACTIVITY_MAP)
    )
    processed["sugar_preference_encoded"] = (
        processed["sugar_preference"].astype(str).str.lower().map(SUGAR_MAP)
    )
    processed = processed.dropna(subset=["activity_level_encoded", "sugar_preference_encoded"])
    return processed


def _profile_to_feature_df(profile: Any) -> pd.DataFrame:
    activity_level = str(getattr(profile, "activity_level", "low")).strip().lower()
    sugar_preference = str(getattr(profile, "sugar_preference", "low")).strip().lower()

    return pd.DataFrame(
        [
            {
                "age": int(getattr(profile, "age", 30)),
                "weight": float(getattr(profile, "weight", 70.0)),
                "activity_level_encoded": ACTIVITY_MAP.get(activity_level, 0),
                "sugar_preference_encoded": SUGAR_MAP.get(sugar_preference, 0),
            }
        ]
    )




def _build_logistic_regression() -> LogisticRegression:
    """Build a LogisticRegression model compatible across sklearn versions."""
    candidate_params = {"max_iter": 1000, "multi_class": "auto"}
    supported_params = inspect.signature(LogisticRegression).parameters
    filtered_params = {
        key: value for key, value in candidate_params.items() if key in supported_params
    }
    return LogisticRegression(**filtered_params)


def train_model(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Train and persist the logistic regression model."""
    df = pd.read_csv(dataset_path)
    processed = _preprocess_dataframe(df)

    X = processed[FEATURE_COLUMNS]
    y = processed[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = _build_logistic_regression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = ModelBundle(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        class_labels=list(model.classes_),
    )
    with model_path.open("wb") as f:
        pickle.dump(bundle, f)

    return {"model_path": str(model_path), "accuracy": accuracy, "n_rows": len(processed)}


def load_model(model_path: Path = MODEL_PATH) -> ModelBundle:
    if not model_path.exists():
        train_model(model_path=model_path)
    with model_path.open("rb") as f:
        return pickle.load(f)


def predict(profile: Any, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Predict diet label for a profile and return probabilities."""
    bundle = load_model(model_path=model_path)
    features = _profile_to_feature_df(profile)

    pred_label = bundle.model.predict(features)[0]
    proba = bundle.model.predict_proba(features)[0]
    probabilities = {
        label: float(prob) for label, prob in sorted(zip(bundle.class_labels, proba), key=lambda x: x[1], reverse=True)
    }

    return {
        "diet_label": str(pred_label),
        "probabilities": probabilities,
        "features": features.iloc[0].to_dict(),
    }
