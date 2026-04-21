"""Simple rule-based nutrition recommender for MVP."""

from dataclasses import dataclass


@dataclass
class UserProfile:
    age: int
    weight: float
    activity_level: str
    dietary_preference: str = "balanced"


def generate_recommendation(profile: UserProfile) -> str:
    """Return a minimal text diet suggestion based on simple rules."""
    activity = profile.activity_level.strip().lower()
    preference = profile.dietary_preference.strip().lower()

    if activity in {"high", "very active"}:
        activity_plan = "Higher-energy meal plan with lean protein, whole grains, and healthy fats"
    elif activity in {"moderate", "medium"}:
        activity_plan = "Balanced meal plan with moderate carbohydrates, vegetables, and lean protein"
    else:
        activity_plan = "Lower-calorie meal plan focused on vegetables, protein, and fiber-rich foods"

    if profile.weight >= 90:
        weight_note = "with portion control and reduced sugary snacks"
    elif profile.weight < 55:
        weight_note = "with nutrient-dense snacks between meals"
    else:
        weight_note = "with regular meal timing"

    age_note = "and calcium-rich foods for bone health" if profile.age >= 50 else "and plenty of hydration"

    if preference in {"vegetarian", "vegan"}:
        preference_note = f"Adapted to a {preference} pattern using legumes, tofu, nuts, and seeds"
    elif preference in {"low carb", "low-carb"}:
        preference_note = "Adjusted for lower carbohydrate intake with extra non-starchy vegetables"
    else:
        preference_note = "Standard balanced dietary pattern"

    return f"{activity_plan}, {weight_note} {age_note}. {preference_note}."
