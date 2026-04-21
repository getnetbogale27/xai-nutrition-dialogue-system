"""SHAP helper utilities for nutrition model predictions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import shap

from src.recommender.ml_model import FEATURE_COLUMNS, DATASET_PATH, _preprocess_dataframe, load_model

HUMAN_FEATURE_NAMES = {
    "age": "age",
    "weight": "weight",
    "activity_level_encoded": "activity_level",
    "sugar_preference_encoded": "sugar_preference",
}


def explain_prediction(profile: Any) -> dict[str, Any]:
    """Generate SHAP contributions for the predicted class."""
    bundle = load_model()

    raw_df = pd.read_csv(DATASET_PATH)
    background = _preprocess_dataframe(raw_df)[FEATURE_COLUMNS].sample(n=min(100, len(raw_df)), random_state=42)

    input_df = pd.DataFrame([
        {
            "age": int(getattr(profile, "age", 30)),
            "weight": float(getattr(profile, "weight", 70.0)),
            "activity_level_encoded": {"low": 0, "medium": 1, "moderate": 1, "high": 2, "very active": 2}.get(str(getattr(profile, "activity_level", "low")).lower(), 0),
            "sugar_preference_encoded": {"low": 0, "high": 1}.get(str(getattr(profile, "sugar_preference", "low")).lower(), 0),
        }
    ])

    explainer = shap.LinearExplainer(bundle.model, background)
    shap_values = explainer.shap_values(input_df)

    prediction = bundle.model.predict(input_df)[0]
    class_index = list(bundle.model.classes_).index(prediction)

    # Multi-class output can be list[class][sample, feature] or array[sample, feature, class]
    if isinstance(shap_values, list):
        values_for_class = shap_values[class_index][0]
    else:
        values_for_class = shap_values[0, :, class_index]

    contributions = []
    readable = []
    for feature, value in zip(FEATURE_COLUMNS, values_for_class):
        v = float(value)
        human_name = HUMAN_FEATURE_NAMES.get(feature, feature)
        contributions.append({"feature": human_name, "contribution": v})
        sign = "+" if v >= 0 else ""
        readable.append(f"{human_name} → {sign}{v:.3f} influence on {prediction}")

    contributions.sort(key=lambda row: abs(row["contribution"]), reverse=True)

    return {
        "predicted_label": str(prediction),
        "contributions": contributions,
        "human_readable": readable,
    }
