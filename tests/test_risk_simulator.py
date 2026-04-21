from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_simulator import BaselineWeightedScoringEngine, NutritionRiskSimulator, RiskProfile


def _profile(**overrides):
    base = dict(
        sugar_intake_g=40.0,
        fiber_intake_g=18.0,
        fruit_veg_servings=3.0,
        processed_food_servings=3.0,
        activity_level="medium",
        sleep_quality="average",
        smoking_status="non_smoker",
        alcohol_use="moderate",
        stress_level=5,
    )
    base.update(overrides)
    return RiskProfile(**base)


def test_baseline_engine_generates_ranked_contributions():
    result = BaselineWeightedScoringEngine().score(_profile())
    assert 0.0 <= result.score <= 100.0
    assert len(result.contributions) == 9
    magnitudes = [abs(c.contribution) for c in result.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_scenario_simulation_updates_score():
    simulator = NutritionRiskSimulator()
    baseline = _profile(sugar_intake_g=80.0)
    output = simulator.simulate_one_variable(baseline, "sugar_intake_g", 20.0)
    assert output["scenario"].score > output["baseline"].score
    assert "sugar intake g" in output["comparison_text"].lower()
