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
    health_goal: str = "general_wellness"
    allergies: tuple[str, ...] = ()
    excluded_foods: tuple[str, ...] = ()
    medical_conditions: tuple[str, ...] = ()


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


def _contains_any_terms(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _build_personalized_meal_ideas(profile: UserProfile, label: str) -> list[str]:
    base_ideas = {
        "balanced": [
            "Mediterranean quinoa bowl with cucumber, tomato, olive oil, and grilled tofu/chicken",
            "Oatmeal with berries, chia seeds, and unsweetened yogurt",
            "Lentil and vegetable soup with a side salad",
        ],
        "high_protein": [
            "Egg/bean scramble with spinach and avocado",
            "Greek yogurt bowl with nuts and pumpkin seeds",
            "Grilled fish or tofu, roasted vegetables, and a small serving of brown rice",
        ],
        "low_carb": [
            "Zucchini noodle stir-fry with lean protein",
            "Large leafy salad with olive oil, nuts, and grilled tofu/chicken",
            "Cauliflower rice bowl with mixed vegetables and tempeh",
        ],
        "low_calorie": [
            "Vegetable soup with chickpeas and herbs",
            "Steamed vegetables with lean protein and a citrus dressing",
            "Berry and spinach smoothie with unsweetened milk alternative",
        ],
    }
    ideas = list(base_ideas.get(label, base_ideas["balanced"]))

    if "blood sugar control" in profile.health_goal:
        ideas.append("High-fiber plate: beans, non-starchy vegetables, and healthy fats")
    if "muscle gain" in profile.health_goal:
        ideas.append("Post-workout plate with lean protein, legumes, and complex carbs")
    if "weight loss" in profile.health_goal:
        ideas.append("Portion-focused bowl with half vegetables, quarter protein, quarter whole grain")

    blocked_terms = {item.strip().lower() for item in (*profile.allergies, *profile.excluded_foods) if item.strip()}
    if "diabetes" in {condition.lower() for condition in profile.medical_conditions}:
        blocked_terms.update({"juice", "sugary", "sweetened"})

    filtered = [idea for idea in ideas if not _contains_any_terms(idea, blocked_terms)]
    return filtered[:3] if filtered else ["No safe default meal idea matched the current restrictions."]


def _personalization_message(profile: UserProfile) -> str:
    goal = profile.health_goal.replace("_", " ")
    conditions = ", ".join(profile.medical_conditions) if profile.medical_conditions else "none reported"
    allergies = ", ".join(profile.allergies) if profile.allergies else "none reported"
    return (
        f"Personalization active: aligned for goal '{goal}', conditions ({conditions}), "
        f"and allergies/restrictions ({allergies})."
    )


def generate_recommendation(profile: UserProfile, mode: str = "ml-based") -> dict:
    """Return recommendation payload for selected mode (`ml-based` default)."""
    mode = (mode or "ml-based").strip().lower()

    if mode == "rule-based":
        label = _rule_based_recommendation(profile)
        meal_ideas = _build_personalized_meal_ideas(profile, label)
        return {
            "mode": "rule-based",
            "diet_label": label,
            "recommendation_text": _label_to_text(label),
            "probabilities": {},
            "meal_ideas": meal_ideas,
            "personalization_message": _personalization_message(profile),
        }

    ml_result = ml_predict(profile)
    label = ml_result["diet_label"]
    meal_ideas = _build_personalized_meal_ideas(profile, label)
    return {
        "mode": "ml-based",
        "diet_label": label,
        "recommendation_text": _label_to_text(label),
        "probabilities": ml_result.get("probabilities", {}),
        "meal_ideas": meal_ideas,
        "personalization_message": _personalization_message(profile),
        "ai_reasoning": {
            "top_probability": max(ml_result.get("probabilities", {}).values(), default=0.0),
            "features_used": ml_result.get("features", {}),
        },
    }
