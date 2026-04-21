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
        base = "Higher-energy meal plan with lean protein, whole grains, and healthy fats"
    elif activity in {"moderate", "medium"}:
        base = "Balanced meal plan with moderate carbohydrates, vegetables, and lean protein"
    else:
        base = "Lower-calorie meal plan focused on vegetables, protein, and fiber-rich foods"

    if profile.weight >= 90:
        weight_note = "with portion control and reduced sugary snacks"
    elif profile.weight < 55:
        weight_note = "with nutrient-dense snacks between meals"
    else:
        weight_note = "with regular meal timing"

    if profile.age >= 50:
        age_note = "and calcium-rich foods for bone health"
    else:
        age_note = "and plenty of hydration"

    if preference in {"vegetarian", "vegan"}:
        pref_note = f"Adapted to a {preference} pattern using legumes, tofu, nuts, and seeds"
    elif preference in {"low carb", "low-carb"}:
        pref_note = "Adjusted for lower carbohydrate intake with extra non-starchy vegetables"
    else:
        pref_note = "Standard balanced dietary pattern"

    return f"{base}, {weight_note} {age_note}. {pref_note}."
