"""Explanation engine that combines prediction and SHAP contributions."""

from __future__ import annotations

from typing import Any

from src.explainability.shap_explainer import explain_prediction
from src.recommender.rules import UserProfile


def _rule_trace(profile: UserProfile, diet_label: str) -> dict[str, Any]:
    return {
        "mode": "rule-based",
        "predicted_label": diet_label,
        "contributions": [
            {"feature": "activity_level", "reason": f"activity_level={profile.activity_level}"},
            {"feature": "weight", "reason": f"weight={profile.weight}"},
            {"feature": "dietary_preference", "reason": f"dietary_preference={profile.dietary_preference}"},
            {"feature": "sugar_preference", "reason": f"sugar_preference={profile.sugar_preference}"},
        ],
    }


def generate_explanation(profile: UserProfile, recommendation: dict) -> dict[str, Any]:
    """Return natural language explanation + structured reasoning trace."""
    mode = recommendation.get("mode", "ml-based")
    predicted_label = recommendation.get("diet_label", "balanced")

    if mode == "ml-based":
        shap_payload = explain_prediction(profile)
        top_features = shap_payload["contributions"][:3]
        highlights = ", ".join(
            f"{item['feature']} ({item['contribution']:+.2f})" for item in top_features
        )
        natural = (
            f"The ML model predicted '{predicted_label}'. "
            f"Top contributing features were {highlights}. "
            f"Positive values increased confidence in this label, while negative values reduced it."
        )

        trace = {
            "mode": "ml-based",
            "predicted_label": predicted_label,
            "contributions": shap_payload["contributions"],
            "human_readable": shap_payload["human_readable"],
            "probabilities": recommendation.get("probabilities", {}),
        }
        return {"natural_language": natural, "trace": trace}

    natural = (
        f"The rule-based engine predicted '{predicted_label}' using activity, weight, "
        f"dietary preference, and sugar preference conditions."
    )
    return {
        "natural_language": natural,
        "trace": _rule_trace(profile, predicted_label),
    }


def generate_counterfactual_explanation(original_input: Any, new_input: Any) -> dict[str, Any]:
    """Generate a human-readable comparison between baseline and modified scenarios."""

    def _read(obj: Any, field: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    tracked_fields = ("age", "weight", "activity_level", "sugar_preference")
    label_map = {
        "age": "Age",
        "weight": "Weight",
        "activity_level": "Activity level",
        "sugar_preference": "Sugar preference",
    }

    key_differences: list[str] = []
    expected_effect: list[str] = []
    for field in tracked_fields:
        before = _read(original_input, field)
        after = _read(new_input, field)
        if before != after:
            key_differences.append(f"{label_map[field]}: {before} → {after}")
            if field == "activity_level":
                expected_effect.append(
                    "Changing activity level typically has a strong directional impact in SHAP analysis."
                )
            elif field == "weight":
                expected_effect.append(
                    "Weight shifts can substantially change calorie-oriented recommendations."
                )
            elif field == "sugar_preference":
                expected_effect.append(
                    "Sugar preference alters low-carb vs balanced tendencies in the model."
                )
            else:
                expected_effect.append("Age contributes a moderate adjustment to model confidence.")

    if not key_differences:
        return {
            "key_differences": ["No input features were changed."],
            "expected_effect": ["Prediction is expected to remain stable."],
            "human_readable": "No counterfactual changes were applied, so the recommendation should stay the same.",
        }

    narrative = (
        "Counterfactual scenario updated the profile by: "
        + "; ".join(key_differences)
        + ". Expected model effect: "
        + " ".join(expected_effect)
    )

    return {
        "key_differences": key_differences,
        "expected_effect": expected_effect,
        "human_readable": narrative,
    }
