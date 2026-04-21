"""Simple Bayesian-style probability layer for interpretable diet reasoning."""

from __future__ import annotations

from typing import Any

DIET_LABELS = ("balanced", "high_protein", "low_carb", "low_calorie")


def _read(input_data: Any, field: str, default: Any = None) -> Any:
    if isinstance(input_data, dict):
        return input_data.get(field, default)
    return getattr(input_data, field, default)


def _age_bucket(age: int | float) -> str:
    if age < 30:
        return "low"
    if age <= 50:
        return "medium"
    return "high"


def _weight_bucket(weight: int | float) -> str:
    if weight < 60:
        return "low"
    if weight <= 85:
        return "normal"
    return "high"


def _normalize(probabilities: dict[str, float]) -> dict[str, float]:
    clipped = {label: max(value, 0.01) for label, value in probabilities.items()}
    total = sum(clipped.values()) or 1.0
    return {label: value / total for label, value in clipped.items()}


def compute_diet_probabilities(input_data: Any) -> dict[str, float]:
    """Compute heuristic Bayesian-like probabilities over diet labels."""
    age = float(_read(input_data, "age", 30))
    weight = float(_read(input_data, "weight", 70))
    activity = str(_read(input_data, "activity_level", "medium")).strip().lower()
    sugar = str(_read(input_data, "sugar_preference", "low")).strip().lower()

    age_group = _age_bucket(age)
    weight_group = _weight_bucket(weight)

    # Base prior over diet recommendations.
    scores = {
        "balanced": 0.25,
        "high_protein": 0.25,
        "low_carb": 0.25,
        "low_calorie": 0.25,
    }

    # Activity likelihood adjustments.
    if activity == "high":
        scores["high_protein"] += 0.20
        scores["balanced"] += 0.08
        scores["low_calorie"] -= 0.05
    elif activity == "low":
        scores["low_calorie"] += 0.18
        scores["high_protein"] -= 0.05
    else:
        scores["balanced"] += 0.10

    # Sugar preference likelihood adjustments.
    if sugar == "high":
        scores["low_carb"] += 0.16
        scores["low_calorie"] += 0.12
        scores["balanced"] -= 0.07
    else:
        scores["balanced"] += 0.08

    # Weight likelihood adjustments.
    if weight_group == "high":
        scores["low_calorie"] += 0.22
        scores["low_carb"] += 0.08
        scores["high_protein"] -= 0.03
    elif weight_group == "low":
        scores["high_protein"] += 0.10
        scores["balanced"] += 0.05
    else:
        scores["balanced"] += 0.06

    # Age likelihood adjustments.
    if age_group == "high":
        scores["balanced"] += 0.10
        scores["low_calorie"] += 0.05
    elif age_group == "low":
        scores["high_protein"] += 0.07

    return _normalize(scores)


def generate_bayesian_explanation(input_data: Any, probabilities: dict[str, float]) -> str:
    """Generate a human-readable explanation of Bayesian probability updates."""
    age_group = _age_bucket(float(_read(input_data, "age", 30)))
    weight_group = _weight_bucket(float(_read(input_data, "weight", 70)))
    activity = str(_read(input_data, "activity_level", "medium")).strip().lower()
    sugar = str(_read(input_data, "sugar_preference", "low")).strip().lower()

    key_points: list[str] = []
    if activity == "high":
        key_points.append("High activity raises the probability of a high_protein plan")
    elif activity == "low":
        key_points.append("Low activity pushes probability toward low_calorie")
    else:
        key_points.append("Medium activity keeps the balanced option competitive")

    if weight_group == "high":
        key_points.append("Higher weight increases low_calorie and low_carb likelihood")
    elif weight_group == "low":
        key_points.append("Lower weight supports balanced/high_protein plans")

    if sugar == "high":
        key_points.append("High sugar preference favors low_carb or low_calorie recommendations")
    else:
        key_points.append("Low sugar preference supports a balanced distribution")

    if age_group == "high":
        key_points.append("Older age slightly increases balanced and low_calorie preference")

    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    summary = ", ".join(f"{diet}: {prob:.1%}" for diet, prob in ranked)
    return (
        "Bayesian view: "
        + "; ".join(key_points)
        + ". Final probability distribution -> "
        + summary
        + "."
    )
