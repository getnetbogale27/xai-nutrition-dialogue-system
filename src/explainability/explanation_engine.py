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
