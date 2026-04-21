"""Template-based explanation engine for recommendation outputs."""

from src.recommender.rules import UserProfile


def generate_explanation(profile: UserProfile, recommendation: str) -> str:
    """Explain why a recommendation was produced from rule signals."""
    activity_reason = {
        "high": "you reported a high activity level, so your body likely needs more daily energy",
        "very active": "you reported a very active lifestyle, so extra fuel and protein were prioritized",
        "moderate": "you reported moderate activity, so a balanced macro split was selected",
        "medium": "you reported medium activity, so a balanced macro split was selected",
    }.get(
        profile.activity_level.strip().lower(),
        "your activity level suggests focusing on lower-calorie, high-fiber foods",
    )

    if profile.weight >= 90:
        weight_reason = "your weight input triggered a focus on portion control and lower sugar intake"
    elif profile.weight < 55:
        weight_reason = "your weight input triggered nutrient-dense snacks for additional energy support"
    else:
        weight_reason = "your weight input fits a maintenance-oriented plan with regular meal timing"

    preference = profile.dietary_preference.strip() or "balanced"

    return (
        f"This recommendation was made because {activity_reason}. "
        f"Also, {weight_reason}. "
        f"Your dietary preference ({preference}) was applied when selecting food options. "
        f"Recommended plan: {recommendation}"
    )
