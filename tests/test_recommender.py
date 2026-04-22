from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender.rules import UserProfile, generate_recommendation


def test_recommendation_includes_personalization_fields():
    profile = UserProfile(
        age=34,
        weight=82.0,
        activity_level="medium",
        sugar_preference="low",
        health_goal="weight_loss",
        allergies=("peanut",),
        excluded_foods=("beef",),
        medical_conditions=("diabetes",),
    )

    result = generate_recommendation(profile, mode="rule-based")

    assert "personalization_message" in result
    assert "meal_ideas" in result
    assert "food_image_links" in result
    assert isinstance(result["meal_ideas"], list)
    assert isinstance(result["food_image_links"], list)
    assert result["meal_ideas"]
    assert result["food_image_links"]
    assert "one-size-fits-all" not in result["recommendation_text"].lower()


def test_meal_ideas_filter_allergy_terms():
    profile = UserProfile(
        age=29,
        weight=70.0,
        activity_level="high",
        sugar_preference="low",
        allergies=("fish",),
    )
    result = generate_recommendation(profile, mode="rule-based")
    assert all("fish" not in idea.lower() for idea in result["meal_ideas"])
