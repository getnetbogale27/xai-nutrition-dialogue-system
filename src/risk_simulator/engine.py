"""Scoring engines for the Nutrition Risk Simulator (XAI)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.risk_simulator.models import FactorContribution, RiskProfile, SimulationResult


class RiskScoringEngine(ABC):
    """Interface for score engines; swappable for Bayesian Network implementation later."""

    @abstractmethod
    def score(self, profile: RiskProfile) -> SimulationResult:
        """Compute score + explainable decomposition for the profile."""


class BaselineWeightedScoringEngine(RiskScoringEngine):
    """Simple weighted additive baseline for explainable nutrition-health scoring."""

    def __init__(self) -> None:
        self._weights: dict[str, float] = {
            "sugar_intake_g": 0.28,
            "fiber_intake_g": 0.22,
            "fruit_veg_servings": 0.20,
            "processed_food_servings": 0.18,
            "activity_level": 0.16,
            "sleep_quality": 0.10,
            "smoking_status": 0.17,
            "alcohol_use": 0.11,
            "stress_level": 0.12,
        }

    def score(self, profile: RiskProfile) -> SimulationResult:
        normalized_inputs = self._normalize_inputs(profile)
        contributions = self._compute_contributions(normalized_inputs)

        base_score = 65.0
        score = max(0.0, min(100.0, base_score + sum(c.contribution for c in contributions)))
        risk_band = self._risk_band(score)

        top_positive = [c for c in contributions if c.direction == "protective"]
        top_negative = [c for c in contributions if c.direction == "risk"]
        top_positive.sort(key=lambda c: c.contribution, reverse=True)
        top_negative.sort(key=lambda c: c.contribution)

        positive_text = top_positive[0].factor if top_positive else "none"
        risk_text = top_negative[0].factor if top_negative else "none"
        explanation_text = (
            f"Overall nutrition health score is {score:.1f}/100 ({risk_band}). "
            f"Strongest protective factor: {positive_text.replace('_', ' ')}. "
            f"Strongest risk factor: {risk_text.replace('_', ' ')}."
        )

        ranked = sorted(contributions, key=lambda item: abs(item.contribution), reverse=True)
        return SimulationResult(
            score=score,
            risk_band=risk_band,
            contributions=ranked,
            explanation_text=explanation_text,
        )

    def _normalize_inputs(self, profile: RiskProfile) -> dict[str, float]:
        activity_map = {"low": -1.0, "medium": 0.0, "high": 1.0}
        sleep_map = {"poor": -1.0, "average": 0.0, "good": 1.0}
        smoking_map = {"non_smoker": 1.0, "occasional": -0.2, "regular": -1.0}
        alcohol_map = {"none": 1.0, "moderate": 0.1, "high": -1.0}

        return {
            "sugar_intake_g": self._clamp((35.0 - profile.sugar_intake_g) / 35.0),
            "fiber_intake_g": self._clamp((profile.fiber_intake_g - 20.0) / 20.0),
            "fruit_veg_servings": self._clamp((profile.fruit_veg_servings - 4.0) / 4.0),
            "processed_food_servings": self._clamp((2.0 - profile.processed_food_servings) / 2.0),
            "activity_level": activity_map[profile.activity_level],
            "sleep_quality": sleep_map[profile.sleep_quality],
            "smoking_status": smoking_map[profile.smoking_status],
            "alcohol_use": alcohol_map[profile.alcohol_use],
            "stress_level": self._clamp((5 - profile.stress_level) / 5.0),
        }

    def _compute_contributions(self, normalized_inputs: dict[str, float]) -> list[FactorContribution]:
        contribution_scale = 20.0
        readable = {
            "sugar_intake_g": "Lower sugar intake improves metabolic resilience.",
            "fiber_intake_g": "Fiber supports satiety and glycemic control.",
            "fruit_veg_servings": "Fruit and vegetable diversity boosts micronutrient quality.",
            "processed_food_servings": "Processed foods often raise sodium and sugar load.",
            "activity_level": "Physical activity improves insulin sensitivity and energy balance.",
            "sleep_quality": "Sleep quality affects appetite regulation and recovery.",
            "smoking_status": "Smoking worsens cardiovascular and inflammation risk.",
            "alcohol_use": "Higher alcohol intake can reduce diet quality and recovery.",
            "stress_level": "High stress is linked to poorer dietary adherence.",
        }
        rows: list[FactorContribution] = []
        for key, value in normalized_inputs.items():
            effect = value * self._weights[key] * contribution_scale
            direction = "protective" if effect >= 0 else "risk"
            rows.append(
                FactorContribution(
                    factor=key,
                    contribution=effect,
                    direction=direction,
                    rationale=readable[key],
                )
            )
        return rows

    @staticmethod
    def _risk_band(score: float) -> str:
        if score >= 80:
            return "Low Risk"
        if score >= 60:
            return "Moderate Risk"
        return "Elevated Risk"

    @staticmethod
    def _clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))
