"""Template-based explanation engine for recommendation outputs."""

from src.recommender.rules import UserProfile


def generate_explanation(profile: UserProfile, recommendation: str) -> str:
    """Explain why a recommendation was produced from rule signals."""
    activity_reason = {
        "high": "you reported a high activity level, so your body likely needs more energy",
        "very active": "you reported a very active lifestyle, so extra fuel and protein were prioritized",
        "moderate": "you reported moderate activity, so a balanced macro split was selected",
        "medium": "you reported medium activity, so a balanced macro split was selected",
    }.get(
        profile.activity_level.strip().lower(),
        "your activity level suggests focusing on lower-calorie, high-fiber foods",
    )

    weight_reason = (
        "Because your weight is on the higher side, the plan emphasizes portion control and lower sugar intake."
        if profile.weight >= 90
        else "Because your weight is in a lower range, the plan includes nutrient-dense snacks to support energy."
        if profile.weight < 55
        else "Because your weight is in a mid-range, the plan keeps regular meal timing and balance."
    )

    preference_reason = (
        f"Your {profile.dietary_preference} preference was included in food choices."
        if profile.dietary_preference
        else "A general balanced preference was used."
    )

    return (
        f"This recommendation was made because {activity_reason}. "
        f"{weight_reason} {preference_reason} "
        f"Recommended plan: {recommendation}"
    )
