"""Nutrition recommender that supports rule-based and ML-based strategies."""

from dataclasses import dataclass

from src.recommender.ml_model import predict as ml_predict


@dataclass
class UserProfile:
    age: int
    weight: float
    activity_level: str
    dietary_preference: str = "balanced"
    sugar_preference: str = "low"


def _rule_based_recommendation(profile: UserProfile) -> str:
    activity = profile.activity_level.strip().lower()
    preference = profile.dietary_preference.strip().lower()

    if activity in {"high", "very active"}:
        activity_plan = "high_protein"
    elif activity in {"moderate", "medium"}:
        activity_plan = "balanced"
    else:
        activity_plan = "low_calorie"

    if profile.weight >= 90:
        return "low_calorie"
    if preference in {"low carb", "low-carb"} or profile.sugar_preference == "high":
        return "low_carb"
    return activity_plan


def _label_to_text(label: str) -> str:
    mapping = {
        "balanced": "Balanced meal plan with vegetables, whole grains, and lean protein.",
        "high_protein": "Higher-protein meal plan with lean meats/plant protein and complex carbs.",
        "low_carb": "Lower-carb meal plan emphasizing non-starchy vegetables and protein.",
        "low_calorie": "Lower-calorie meal plan with portion control and high-fiber foods.",
    }
    return mapping.get(label, "Balanced meal plan with whole-food ingredients.")


def generate_recommendation(profile: UserProfile, mode: str = "ml-based") -> dict:
    """Return recommendation payload for selected mode (`ml-based` default)."""
    mode = (mode or "ml-based").strip().lower()

    if mode == "rule-based":
        label = _rule_based_recommendation(profile)
        return {
            "mode": "rule-based",
            "diet_label": label,
            "recommendation_text": _label_to_text(label),
            "probabilities": {},
        }

    ml_result = ml_predict(profile)
    label = ml_result["diet_label"]
    return {
        "mode": "ml-based",
        "diet_label": label,
        "recommendation_text": _label_to_text(label),
        "probabilities": ml_result.get("probabilities", {}),
    }
