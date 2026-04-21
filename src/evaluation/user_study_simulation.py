"""Simulation utilities for lightweight user-study style evaluation."""

from __future__ import annotations

from typing import Any


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def simulate_user_feedback(explanation: dict[str, Any], recommendation: dict[str, Any]) -> dict[str, float]:
    """Simulate perceived trust, explanation quality, and satisfaction.

    Scoring heuristic:
    - ML explanations are scored higher for trust/quality.
    - Rule-based explanations are scored in the medium range.
    - Missing or unclear explanations are scored lower.
    """
    mode = str(recommendation.get("mode", "")).strip().lower()
    natural_language = str(explanation.get("natural_language", "")).strip()
    trace = explanation.get("trace", {}) if isinstance(explanation, dict) else {}

    has_clear_text = len(natural_language.split()) >= 8
    has_structure = bool(trace.get("human_readable") or trace.get("contributions"))

    if mode == "ml-based":
        trust, quality, satisfaction = 0.82, 0.80, 0.78
    elif mode == "rule-based":
        trust, quality, satisfaction = 0.65, 0.62, 0.64
    else:
        trust, quality, satisfaction = 0.40, 0.38, 0.42

    if not has_clear_text:
        trust -= 0.18
        quality -= 0.20
        satisfaction -= 0.16

    if not has_structure:
        trust -= 0.12
        quality -= 0.14
        satisfaction -= 0.10

    return {
        "trust_score": _clamp(trust),
        "explanation_quality": _clamp(quality),
        "satisfaction": _clamp(satisfaction),
    }
