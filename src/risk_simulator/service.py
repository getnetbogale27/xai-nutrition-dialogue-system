"""Service-layer orchestration for baseline scoring and scenario simulation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.risk_simulator.engine import BaselineWeightedScoringEngine, RiskScoringEngine
from src.risk_simulator.models import RiskProfile, SimulationResult


class NutritionRiskSimulator:
    """High-level API that wraps scoring engine selection and scenario simulation."""

    def __init__(self, engine: RiskScoringEngine | None = None) -> None:
        self.engine = engine or BaselineWeightedScoringEngine()

    def evaluate(self, profile: RiskProfile) -> SimulationResult:
        return self.engine.score(profile)

    def simulate_one_variable(self, profile: RiskProfile, variable: str, new_value: Any) -> dict[str, SimulationResult | str]:
        baseline = self.evaluate(profile)
        scenario_profile = replace(profile, **{variable: new_value})
        scenario = self.evaluate(scenario_profile)
        delta = scenario.score - baseline.score
        direction = "improved" if delta >= 0 else "worsened"
        comparison_text = (
            f"Changing {variable.replace('_', ' ')} from {getattr(profile, variable)} to {new_value} "
            f"{direction} the score by {delta:+.1f} points."
        )
        return {
            "baseline": baseline,
            "scenario": scenario,
            "comparison_text": comparison_text,
        }
