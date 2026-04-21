"""Domain models for the Nutrition Risk Simulator (XAI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActivityLevel = Literal["low", "medium", "high"]
SleepQuality = Literal["poor", "average", "good"]
SmokingStatus = Literal["non_smoker", "occasional", "regular"]
AlcoholUse = Literal["none", "moderate", "high"]


@dataclass(frozen=True)
class RiskProfile:
    """Input profile used by the scoring engines."""

    sugar_intake_g: float
    fiber_intake_g: float
    fruit_veg_servings: float
    processed_food_servings: float
    activity_level: ActivityLevel
    sleep_quality: SleepQuality
    smoking_status: SmokingStatus
    alcohol_use: AlcoholUse
    stress_level: int


@dataclass(frozen=True)
class FactorContribution:
    """Per-factor signed contribution to the final score."""

    factor: str
    contribution: float
    direction: Literal["protective", "risk"]
    rationale: str


@dataclass(frozen=True)
class SimulationResult:
    """Main output for baseline scoring and future engine variants."""

    score: float
    risk_band: str
    contributions: list[FactorContribution]
    explanation_text: str
